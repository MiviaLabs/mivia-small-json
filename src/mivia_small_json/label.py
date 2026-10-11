# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Weak-label tiers for the improvement pool.

Only labeled development data may set a weak-label threshold, decide a
label, or admit a record. Frozen audit data never does.

- **Tier 1 — model and rules agree.** Off unless a calibration is
  recorded for the router in the tree. The probability must sit within
  ``CALIBRATION_MAX_ERROR`` of the observed rate on development data.
  That error is measured over the range tier 1 admits. The record also
  carries a reliability measure (Brier and ECE) and the admission
  threshold chosen from a precision-at-coverage target, so confidence
  is a calibrated number rather than the raw max-probability. See
  :func:`measure_dev_calibration` and :func:`precision_at_coverage`.
- **Tier 2 — rules-dominant.** Runs only for a class the rules layer
  earns on development data: normalized margin >= ``RULES_DOM_MARGIN``
  and the class is trusted for that vote.
- **Hard pool.** Everything else. Review only. Never trained.

Trust is measured, not assumed. :func:`measure_dev_rules` scores every
class from one pooled confusion matrix built from the labeled
development rows in :data:`DEV_REFERENCE`. One matrix scores every class,
so a false positive sitting in another class's rows still counts against
the class the rules voted for.

A class with fewer than ``MIN_DEV_SUPPORT`` labeled development rows has
unknown trust. Its numbers are still reported. An unknown class never
admits a tier-2 label. Trust is never inferred from frozen audit data.

The trust and the calibration are measured on a FROZEN development
reference (:data:`DEV_REFERENCE`), a committed snapshot of the labeled
development rows, not on the growing ``data/real/*/dev.jsonl`` files.
Admitting real dev rows once re-baselined ``measure_dev_rules`` (n
2800 -> 5093, ``trusted_classes`` -> ``[]``) and silently switched the
weak-label flywheel off. The reference pins the basis instead.

The trade-off is deliberate: the trust now describes the frozen snapshot
and no longer tracks later dev growth. That is the point — a stable
basis — and re-recording the reference is a deliberate step
(:func:`record_dev_reference`), not something a data top-up does behind
the measurement's back. The reference is development data (the same rows
and labels the dev files carried), so the frozen-audit refusal in
:func:`check_calibration` does not apply to it, even though it lives in
the frozen tree for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from . import bootstrap
from .registry import active

RULES_TRUST_F1 = 0.85
MIN_DEV_SUPPORT = 20
AGREE_MIN_PROBA = 0.90
RULES_DOM_MARGIN = 0.50

# A calibration is accepted only when the router's probability sits this
# close to the observed rate on development data. The limit was fixed
# before the measurement was taken.
CALIBRATION_MAX_ERROR = 0.10

# The number of development rows needed behind one admitted range.
CALIBRATION_MIN_ROWS = 100

# The fraction of development rows tier 1 is willing to auto-admit. The
# admission threshold is the lowest confidence in that most-confident
# slice, so the calibrated threshold can only tighten ``AGREE_MIN_PROBA``,
# never loosen it below the range the bands scored.
ADMISSION_TARGET_COVERAGE = 0.95

# The admitted range is cut into equal bands and each is scored on its
# own. A pooled mean lets a bad tail cancel against a good head.
CALIBRATION_BINS = 10

# Two standard errors. A band whose prediction sits inside the interval
# its own observed rate supports is not distinguishably miscalibrated,
# however wide that interval is on a thin band.
CALIBRATION_Z = 1.959964

# Where a measured calibration is recorded.
CALIBRATION_PATH = "data/calibration.json"
CAPS = {"per_class_cycle": 150, "hard_cycle": 100, "log_event_share": 0.25}

# A deterministic sample of auto-labeled rows is spot-checked by a human, so
# weak-label precision is measured over time instead of assumed.
AUDIT_RATE = 0.05
AUDIT_SEED = 42

# The labeled development files, the source a fresh reference is recorded
# from. The second item is the gold schema for every row in that file. A
# value of None means the gold schema comes from each row's own "schema"
# key. The trust and the calibration do NOT read these growing files; they
# read :data:`DEV_REFERENCE`, the frozen snapshot recorded from them.
DEV_SOURCES = (
    ("data/synthetic/dev.jsonl", None),
    ("data/real/issue/dev.jsonl", "issue"),
    ("data/real/commit/dev.jsonl", "commit"),
    ("data/real/log_event/dev.jsonl", "log_event"),
)

# The frozen development reference: a committed, one-row-per-dev-record
# snapshot of the labeled development rows, in order. It is the fixed
# basis :func:`dev_rows` reads, so admitting rows to the growing dev files
# cannot re-baseline the trust or the calibration. It lives in the frozen
# tree for reproducibility but is development data, not a frozen audit
# set. Re-record it deliberately with :func:`record_dev_reference`.
DEV_REFERENCE = "data/heldout/dev-reference.jsonl"

# A calibration source must name a development split, and must not name a
# frozen audit set. This is an allow-list, so an unfamiliar source is
# refused rather than trusted.
FROZEN_MARKERS = ("wild", "heldout", "test.jsonl", "frozen", "audit")
DEV_MARKERS = ("dev",)


@dataclass(frozen=True)
class DevCalibration:
    """A probability calibration measured on labeled development data.

    ``source`` names the development split the calibration was measured
    on. ``n`` is its labeled row count. ``max_abs_error`` is the largest
    gap that is both over the limit and statistically distinguishable,
    taken over the probability bands the tier admits. A pooled mean is
    not a bound and must not be used as one.

    A band only counts against the limit when its prediction lies
    outside the interval its own observed rate supports, so a thin band
    cannot fail the record on noise.

    ``worst_band_error`` carries the stored record's worst band gap so a
    report can show it. No gate reads it.

    ``admission_threshold`` is the calibrated probability tier 1 now
    requires, chosen from a precision-at-coverage target on development
    data. It never sits below ``AGREE_MIN_PROBA``: the bands were cut over
    ``p >= AGREE_MIN_PROBA``, so admitting under that floor would admit
    rows no band scored. A record written before this field existed reads
    back with the floor as its threshold.
    """

    source: str
    n: int
    max_abs_error: float = 0.0
    worst_band_error: float | None = None
    admission_threshold: float = AGREE_MIN_PROBA


def _outside_interval(hits: int, count: int, claimed: float) -> bool:
    """True when ``claimed`` sits outside the interval the band supports.

    A thin band cannot pin its own rate down. Twelve rows that are all
    correct support a rate whose interval reaches well below one, so a
    prediction of 0.90 inside that interval is not evidence that the
    band is badly calibrated.
    """
    if count == 0:
        return True
    z = CALIBRATION_Z
    rate = hits / count
    denom = 1.0 + z * z / count
    centre = (rate + z * z / (2 * count)) / denom
    half = (z * math.sqrt(rate * (1.0 - rate) / count
                          + z * z / (4.0 * count * count))) / denom
    return not (centre - half <= claimed <= centre + half)


def _ranked_scores(router, pairs) -> list[tuple[float, int]]:
    """``(confidence, hit)`` per pair, ranked by confidence descending.

    ``confidence`` is the router's top-class probability; ``hit`` is 1
    when that top class is the gold schema. One pass over the pairs, so
    the band measurement and the coverage threshold share the scores.
    """
    classes = list(router.named_steps["clf"].classes_)
    scored: list[tuple[float, int]] = []
    for text, gold in pairs:
        proba = router.predict_proba([text])[0]
        top = int(proba.argmax())
        scored.append((float(proba[top]), int(classes[top] == gold)))
    scored.sort(key=lambda row: row[0], reverse=True)
    return scored


def _threshold_at_coverage(scored, coverage) -> tuple[float, float]:
    """Lowest admitted confidence and the precision there, for a coverage.

    ``scored`` is ranked descending by confidence. Admits the most
    confident ``coverage`` fraction and reports the lowest admitted
    confidence (the threshold) and the fraction correct among the
    admitted rows (precision).
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must sit in (0, 1]")
    if not scored:
        raise ValueError("precision_at_coverage needs development pairs")
    admitted = max(1, math.ceil(coverage * len(scored)))
    head = scored[:admitted]
    threshold = head[-1][0]
    precision = sum(hit for _, hit in head) / len(head)
    return threshold, precision


def precision_at_coverage(router, pairs, coverage) -> tuple[float, float]:
    """Probability threshold that admits ``coverage`` of dev pairs, and precision.

    Ranks every development pair by the router's top-class probability,
    admits the most confident ``coverage`` fraction, and returns the
    lowest admitted probability (the threshold) plus the fraction of
    admitted rows whose top class matches the gold schema (precision).

    Development data only. A threshold set from frozen audit data would
    leak the audit set into admission, so ``pairs`` must come from
    :func:`dev_rows`. Read-only, deterministic, offline.
    """
    if router is None:
        raise ValueError("precision_at_coverage needs the router")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("precision_at_coverage needs development pairs")
    return _threshold_at_coverage(_ranked_scores(router, pairs), coverage)


def measure_dev_calibration(root=None, router=None,
                            threshold: float = AGREE_MIN_PROBA,
                            target_coverage: float = ADMISSION_TARGET_COVERAGE
                            ) -> dict:
    """Measure how far the router's probability sits from the truth.

    Measures only the range tier 1 admits, which is ``p >= threshold``.
    A bin below the threshold never admits a record, so its error does
    not inform this decision.

    Cuts that range into equal bands and scores each on its own. The
    largest band deviation is the bound. Pooling the whole range would
    let a badly calibrated tail cancel against a confident head, and the
    result would not be a bound at all.

    The record also carries a reliability measure over the admitted
    range: ``brier`` (the mean squared gap between confidence and the
    0/1 outcome) and ``ece`` (the band gaps averaged by band size). Both
    are reported, neither gates.

    ``admission_threshold`` is chosen from a precision-at-coverage target
    on the same development scores, but never below ``threshold``: the
    bands scored ``p >= threshold``, so admitting under it would admit
    rows no band measured. The tier reads this threshold instead of the
    fixed ``AGREE_MIN_PROBA``.

    Returns the record to store at :data:`CALIBRATION_PATH`. Raises
    ValueError when the sample is too small or a band is too far off.
    Read-only, deterministic, offline.
    """
    pairs, files = dev_rows(root)
    if router is None:
        raise ValueError("measure_dev_calibration needs the router")
    if threshold >= 1.0:
        raise ValueError("threshold must leave a non-empty probability range")
    scored = _ranked_scores(router, pairs)
    width = (1.0 - threshold) / CALIBRATION_BINS
    bands: dict[int, list[tuple[float, int]]] = {}
    seen = 0
    correct = 0
    proba_total = 0.0
    brier_total = 0.0
    for confidence, hit in scored:
        if confidence < threshold:
            continue
        seen += 1
        proba_total += confidence
        brier_total += (confidence - hit) ** 2
        correct += hit
        index = min(CALIBRATION_BINS - 1,
                    int((confidence - threshold) / width))
        bands.setdefault(index, []).append((confidence, hit))
    if seen < CALIBRATION_MIN_ROWS:
        raise ValueError(f"only {seen} development rows reach p >= {threshold}; "
                         f"need {CALIBRATION_MIN_ROWS}")
    # The admission threshold is chosen from the same scores, clamped to
    # the measured floor so it can only tighten the fixed floor. The
    # precision is recomputed at the clamped threshold, which is the
    # threshold tier 1 actually applies.
    chosen, _ = _threshold_at_coverage(scored, target_coverage)
    admission_threshold = max(threshold, chosen)
    admitted = [hit for confidence, hit in scored
                if confidence >= admission_threshold]
    admission_precision = (sum(admitted) / len(admitted)) if admitted else 0.0
    admission_coverage = (len(admitted) / len(scored)) if scored else 0.0
    predicted = proba_total / seen
    observed = correct / seen
    brier = brier_total / seen
    ece = 0.0
    worst = 0.0
    worst_band = None
    significant = 0.0
    significant_band = None
    for index in sorted(bands):
        # Every band is scored. Skipping a thin one would leave a hole
        # exactly one row wide, and the bias falls on the lowest
        # confidence admitted rows because a well calibrated router
        # under-states its own confidence there.
        entries = bands[index]
        mean_p = sum(e[0] for e in entries) / len(entries)
        hits = sum(e[1] for e in entries)
        gap = abs(mean_p - hits / len(entries))
        ece += (len(entries) / seen) * gap
        if gap > worst:
            worst = gap
            worst_band = index
        # A band counts against the limit only when its prediction lies
        # outside the interval its own observed rate supports. That keeps
        # a thin band from failing the record on noise.
        if (gap > CALIBRATION_MAX_ERROR
                and _outside_interval(hits, len(entries), mean_p)
                and gap > significant):
            significant = gap
            significant_band = index
    # `significant` is only set for a band already over the limit, so the
    # check below stands on its own and does not lean on that invariant.
    if significant > 0:
        raise ValueError(f"calibration band {significant_band} is off by "
                         f"{significant:.4f}, above {CALIBRATION_MAX_ERROR:.4f}")
    return {
        "source": "+".join(entry["path"] for entry in files),
        "files": [entry["sha256"] for entry in files],
        "n": seen,
        "min_probability": threshold,
        "admission_threshold": round(admission_threshold, 4),
        "admission_coverage": round(admission_coverage, 4),
        "admission_precision": round(admission_precision, 4),
        "bands_scored": len(bands),
        "predicted": round(predicted, 4),
        "observed_accuracy": round(observed, 4),
        "mean_abs_error": round(abs(predicted - observed), 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "max_abs_error": round(worst, 4),
        "max_abs_error_band": worst_band,
        "max_significant_error": round(significant, 4),
        "max_error_allowed": CALIBRATION_MAX_ERROR,
    }


def _optional_float(value) -> float | None:
    """A reported field; unreadable means absent, not refused."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def calibration_from_disk(root=None, sidecar=None) -> DevCalibration | None:
    """Read a measured calibration, or None when none applies.

    A calibration is valid only for the artifact it was measured on. A
    retrain changes the router hash, so a stale record is ignored and
    tier 1 stays off until someone measures the new artifact again.
    """
    root = Path(root) if root is not None else Path(".")
    path = root / CALIBRATION_PATH
    from .model import sidecar_path

    sidecar = Path(sidecar) if sidecar is not None else sidecar_path(root)
    if not path.is_file() or not sidecar.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        digest = sidecar.read_text(encoding="utf-8").strip()
        if data["router_sha256"] != digest:
            return None
        # A record written before the field existed keeps the fixed floor
        # as its threshold, which is what it was measured and admitted at.
        admission = _optional_float(data.get("admission_threshold"))
        cand = DevCalibration(source=data["source"], n=int(data["n"]),
                              max_abs_error=float(
                                  data["max_significant_error"]),
                              worst_band_error=_optional_float(
                                  data.get("max_abs_error")),
                              admission_threshold=(AGREE_MIN_PROBA
                                                   if admission is None
                                                   else admission))
        # This reader must agree with check_calibration. Returning a
        # record that later raises would abort the cycle that asked.
        if not _tier1_enabled(cand):
            return None
        if float(data["min_probability"]) != AGREE_MIN_PROBA:
            # The bands were cut over a different range than the one the
            # tier now admits. The hash pins the artifact, not the range.
            return None
        return cand
    except (ValueError, KeyError, TypeError):
        return None


def _is_frozen_audit_source(source: str) -> bool:
    """True when a calibration source names a frozen AUDIT set.

    The frozen development reference is a snapshot of the labeled
    development rows, so it is development data even though it lives in
    the frozen tree. It is exempt; a frozen audit set is not. The exact
    reference token is stripped BEFORE the marker scan, so a source that
    concatenates the reference with a frozen audit path (``dev-reference
    .jsonl+data/wild/issue.jsonl``) is still refused on the remainder.
    """
    lowered = source.lower()
    remainder = lowered.replace(DEV_REFERENCE.lower(), "")
    return any(marker in remainder for marker in FROZEN_MARKERS)


def _is_dev_source(source: str) -> bool:
    """True when a calibration source names a KNOWN development split.

    Path-component aware: a bare ``"dev"`` substring would accept
    ``device-logs.jsonl``. Only the frozen dev reference, a ``dev.jsonl``
    path component (``data/synthetic/dev.jsonl``, ``data/real/*/dev
    .jsonl``), or a promoted-pool file count.
    """
    lowered = source.lower()
    if DEV_REFERENCE.lower() in lowered:
        return True
    parts = re.split(r"[\\/+]+", lowered)
    return "dev.jsonl" in parts or "promoted" in parts


def check_calibration(calibration: DevCalibration | None) -> None:
    """Raise unless ``calibration`` is a usable development measurement.

    ``None`` raises, because an uncalibrated probability must not admit
    a record.
    """
    if calibration is None:
        raise ValueError("tier 1 needs a development calibration result")
    if not isinstance(calibration, DevCalibration):
        raise TypeError("calibration must be DevCalibration, got "
                        f"{type(calibration).__name__}")
    source = str(calibration.source).lower()
    if _is_frozen_audit_source(source):
        raise ValueError("calibration must come from development data, not "
                         f"{calibration.source!r}")
    if not _is_dev_source(source):
        raise ValueError("calibration source must name a development split, "
                         f"got {calibration.source!r}")
    if calibration.n <= 0:
        raise ValueError("calibration needs labeled development rows")
    if calibration.n < CALIBRATION_MIN_ROWS:
        # The module's own floor: measure_dev_calibration refuses to record
        # a measurement over fewer rows, so a stored record claiming less
        # is stale or hand-edited and must not enable tier 1 (V5,
        # 2026-10-10: n=3 shipped and enabled the tier).
        raise ValueError("calibration was measured over "
                         f"{calibration.n} rows; the measurement needs at "
                         f"least {CALIBRATION_MIN_ROWS}")
    if calibration.max_abs_error > CALIBRATION_MAX_ERROR:
        raise ValueError("calibration error "
                         f"{calibration.max_abs_error:.4f} exceeds "
                         f"{CALIBRATION_MAX_ERROR:.4f}")
    # The admission threshold may tighten the fixed floor, never loosen
    # it: the bands were cut over ``p >= AGREE_MIN_PROBA``, so admitting
    # under that floor would admit rows no band scored.
    if not AGREE_MIN_PROBA <= calibration.admission_threshold <= 1.0:
        raise ValueError("calibration admission threshold must sit within "
                         f"[{AGREE_MIN_PROBA:.2f}, 1.0], got "
                         f"{calibration.admission_threshold}")


def _tier1_enabled(calibration: DevCalibration | None) -> bool:
    """True only for a valid development calibration. Never raises."""
    try:
        check_calibration(calibration)
    except (TypeError, ValueError):
        return False
    return True


def _read_dev_file(path: Path) -> tuple[list[dict], str | None]:
    """Parsed rows plus the sha256 of the exact bytes they came from.

    The file is read once. Hashing and parsing share one read, so a
    cycle never loads a development corpus twice.
    """
    if not path.is_file():
        return [], None
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows, hashlib.sha256(raw).hexdigest()


def record_dev_reference(root: Path | None = None) -> Path:
    """Rewrite the frozen development reference from the dev files.

    A deliberate re-record. The reference is the fixed basis the trust
    and the calibration are measured on, so writing a new one re-baselines
    both. Reads :data:`DEV_SOURCES`, resolves each row's gold schema, and
    writes one ``{"raw", "schema"}`` row per dev record, in order.
    Returns the reference path.
    """
    root = Path(root) if root is not None else Path(".")
    rows: list[dict] = []
    for rel, fixed_gold in DEV_SOURCES:
        source_rows, _ = _read_dev_file(root / rel)
        for row in source_rows:
            gold = fixed_gold if fixed_gold is not None else row.get("schema")
            text = row.get("raw")
            if not gold or not text:
                continue
            rows.append({"raw": text, "schema": str(gold)})
    target = root / DEV_REFERENCE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in rows), encoding="utf-8")
    return target


def dev_rows(root: Path | None = None
             ) -> tuple[list[tuple[str, str]], list[dict]]:
    """Labeled ``(text, gold_schema)`` pairs from the frozen reference.

    Reads :data:`DEV_REFERENCE`, the committed snapshot of the labeled
    development rows, so admitting rows to the growing
    ``data/real/*/dev.jsonl`` files cannot re-baseline the trust or the
    calibration. Locally promoted hard rows still join the set (see
    :func:`load_promoted`).

    Returns the pairs and a per-file provenance list carrying the row
    count and the sha256 of the reference. A missing reference
    contributes nothing and raises nothing.
    """
    root = Path(root) if root is not None else Path(".")
    pairs: list[tuple[str, str]] = []
    files: list[dict] = []
    rows, digest = _read_dev_file(root / DEV_REFERENCE)
    added = 0
    for row in rows:
        gold = row.get("schema")
        text = row.get("raw")
        if not gold or not text:
            continue
        pairs.append((text, str(gold)))
        added += 1
    if digest is not None:
        files.append({"path": DEV_REFERENCE, "rows": added, "sha256": digest})
    # Promoted hard rows join the pairs, so their provenance joins the
    # file list too: per file, the path, the usable row count and the
    # sha256 of the exact bytes, so `files` accounts for every row in
    # the returned pairs.
    promoted_dir = root / "data" / "pool" / "promoted"
    if promoted_dir.is_dir():
        for path in sorted(promoted_dir.glob("*.jsonl")):
            raw = path.read_bytes()
            file_added = 0
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                gold = row.get("schema")
                text = row.get("raw")
                if gold and text:
                    pairs.append((text, str(gold)))
                    file_added += 1
            files.append({"path": f"data/pool/promoted/{path.name}",
                          "rows": file_added,
                          "sha256": hashlib.sha256(raw).hexdigest()})
    return pairs, files


def audit_sample(rows: list[dict], rate: float = AUDIT_RATE,
                 seed: int = AUDIT_SEED) -> list:
    """Deterministic sample of row ids for human spot-check. Never raises."""
    import random

    ids = [r.get("id") for r in rows if r.get("id")]
    if not ids or rate <= 0:
        return []
    rng = random.Random(seed)
    n = min(len(ids), max(1, int(len(ids) * rate)))
    return sorted(rng.sample(ids, n))


def load_promoted(root=None) -> list[dict]:
    """Reviewed hard rows promoted into the labeled pool (gitignored).

    A human marks a hard row ``{"review": "accept"}`` and
    ``scripts/promote_hard.py`` copies it here. The rows join the labeled
    development set, closing the otherwise one-way weak-label loop.
    """
    base = (Path(root) if root is not None else Path(".")) / "data" / "pool" / "promoted"
    out: list[dict] = []
    if base.is_dir():
        for path in sorted(base.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
    return out


def measure_dev_rules(root: Path | None = None) -> dict:
    """Complete confusion matrix of the rules layer over the dev sets.

    Every labeled development row votes once. One pooled matrix scores
    every class, so a false positive found in another class's rows
    counts against the class it was voted for.

    A class with fewer than ``MIN_DEV_SUPPORT`` rows is ``unknown``. Its
    precision, recall and F1 are reported, but its trust is not granted.
    Read-only, deterministic, offline, no network.
    """
    bootstrap.register_example_schemas()
    pairs, files = dev_rows(root)
    classes = sorted(set(active()) | {gold for _, gold in pairs})
    confusion = {gold: {pred: 0 for pred in classes} for gold in classes}
    support = {cls: 0 for cls in classes}
    abstentions = {cls: 0 for cls in classes}
    for text, gold in pairs:
        vote, _ = rules_vote(text)
        support[gold] = support.get(gold, 0) + 1
        if vote is None or vote not in confusion[gold]:
            abstentions[gold] = abstentions.get(gold, 0) + 1
            continue
        confusion[gold][vote] += 1
    per_class: dict[str, dict] = {}
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[gold][cls] for gold in classes if gold != cls)
        # Two recall readings, reported side by side. The historical
        # primary credits an abstention to nobody (it is neither hit nor
        # miss), which inflates recall whenever abstentions are common;
        # it still gates trust, per the 2026-10-09 decision, so behavior
        # is unchanged. The honest variant counts every abstention as
        # the miss it is and is reported beside it, never gating.
        fn = support[cls] - tp - abstentions[cls]
        fn_honest = support[cls] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recall_honest = (tp / (tp + fn_honest) if tp + fn_honest else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        f1_honest = (2 * precision * recall_honest
                     / (precision + recall_honest)
                     if precision + recall_honest else 0.0)
        enough = support[cls] >= MIN_DEV_SUPPORT
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "recall_excl_abstentions": round(recall, 4),
            "f1_excl_abstentions": round(f1, 4),
            "recall_incl_abstention_misses": round(recall_honest, 4),
            "f1_incl_abstention_misses": round(f1_honest, 4),
            "support": support[cls],
            "false_positives": fp,
            "abstentions": abstentions[cls],
            "status": ("unknown" if not enough
                       else "trusted" if f1 >= RULES_TRUST_F1
                       else "untrusted"),
        }
    return {
        "source": "development",
        "rule": "complete pooled confusion matrix over labeled development sets",
        "min_support": MIN_DEV_SUPPORT,
        "trust_f1": RULES_TRUST_F1,
        "n": len(pairs),
        "pairs": pairs,
        "files": files,
        "confusion": confusion,
        "per_class": per_class,
        "unknown_classes": sorted(cls for cls, row in per_class.items()
                                  if row["status"] == "unknown"),
        "trusted_classes": sorted(cls for cls, row in per_class.items()
                                  if row["status"] == "trusted"),
    }


def trust_disclosure(measure: dict | None = None) -> dict:
    """Trust source and per-class verdicts, for the flywheel report."""
    measure = measure or measure_dev_rules()
    return {
        "source": measure["source"],
        "rule": measure["rule"],
        "files": [entry["path"] for entry in measure["files"]],
        "min_support": measure["min_support"],
        "trust_f1": measure["trust_f1"],
        "n": measure["n"],
        "unknown_classes": measure["unknown_classes"],
        "trusted_classes": measure["trusted_classes"],
        "per_class": measure["per_class"],
    }


def _trust(measure: dict | None = None) -> dict[str, bool]:
    """Class -> trusted?, from a measured development confusion matrix.

    Fresh by default. A class with unknown trust is never trusted.
    """
    measure = measure or measure_dev_rules()
    return {cls: row.get("status") == "trusted"
            for cls, row in measure.get("per_class", {}).items()}


def rules_vote(text: str) -> tuple[str | None, float]:
    """Return (schema, normalized margin) from the rules routing scores."""
    from . import rules

    scores = rules.route_scores(text)
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, best = ranked[0]
    margin = (best - ranked[1][1]) / best if len(ranked) > 1 and best else 0.0
    return (top if best > 0 else None), margin


def label_record(text: str, router, trust: dict[str, bool] | None = None,
                 calibration: DevCalibration | None = None) -> dict:
    """Assign a pool tier to one raw record. Never raises.

    Tier 1 needs ``calibration``. Without it the model's probability is
    reported but never admits a record, because the probabilities this
    router emits are uncalibrated. With it, admission uses the
    calibration's own ``admission_threshold`` rather than the fixed
    ``AGREE_MIN_PROBA``. Hard rows record the model's vote in ``reason``
    so review sees both sides of a disagreement, not only the rules vote.
    """
    # An empty mapping is a deliberate "trust nothing" policy. Only
    # `None` means "measure it yourself"; `or` would discard the caller's
    # intent and admit labels from a fresh measurement.
    if trust is None:
        trust = _trust()
    tier1 = _tier1_enabled(calibration)
    out = {"raw": text, "tier": "hard", "schema": None, "reason": "no-vote",
           "proba": None, "tier1_enabled": tier1}
    mschema = None
    try:
        rschema, rmargin = rules_vote(text)
        if router is not None:
            proba = router.predict_proba([text])[0]
            classes = list(router.named_steps["clf"].classes_)
            order = proba.argsort()[::-1]
            p1, mschema = float(proba[order[0]]), str(classes[order[0]])
            out["proba"] = round(p1, 4)
            if (tier1 and p1 >= calibration.admission_threshold
                    and mschema == rschema and trust.get(mschema)):
                out.update(tier="confident", schema=mschema,
                           reason="tier1: model+rules agree")
                return out
        if rschema is not None and trust.get(rschema) and rmargin >= RULES_DOM_MARGIN:
            out.update(tier="confident", schema=rschema,
                       reason=f"tier2: rules-dominant margin={rmargin:.2f}")
            return out
        if rschema is None:
            out["reason"] = ("no rules vote" if mschema is None
                             else f"no rules vote model={mschema}")
        elif mschema == rschema and not trust.get(rschema):
            # Agreement alone is not admission: the class's rule trust
            # gates tier 1 the same way it gates tier 2.
            out["reason"] = (f"tier1 blocked: class {rschema} is not trusted "
                             f"for admission")
        else:
            out["reason"] = (f"disagreement model={mschema or 'none'} "
                             f"rules={rschema}")
    except Exception as exc:  # never raise — pool labeling is best-effort
        out["reason"] = f"label-error: {type(exc).__name__}"
    return out


def build_pools(records: list[dict], router,
                trust: dict[str, bool] | None = None,
                calibration: DevCalibration | None = None,
                measure: dict | None = None
                ) -> tuple[list[dict], list[dict], dict]:
    """Split collected records into (confident, hard, stats) with caps.

    Trust is measured on the labeled development sets by default, never
    on frozen audit data. The hard cap bounds the hard POOL, never the
    scan: every record is labeled, so confident rows later in the batch
    are still collected once the hard pool is full.

    ``stats`` also carries the pre-admission view the caps must not hide:
    ``pool_per_class`` counts every confident label BEFORE the caps, and
    ``dropped_over_cap`` names how many confident and hard labels the
    caps dropped. The flywheel's drift gate reads ``pool_per_class``.
    """
    bootstrap.register_example_schemas()  # 4-schema view (pull_request included)
    if measure is None:
        measure = measure_dev_rules()
    if trust is None:
        trust = _trust(measure)
    if calibration is not None:
        check_calibration(calibration)
    confident: list[dict] = []
    hard: list[dict] = []
    per_class: dict[str, int] = {}
    # The PRE-ADMISSION label pool: every record the weak labeler marked
    # confident, counted BEFORE the caps thin it. The caps are admission
    # policy, not evidence about the world — the flywheel's drift gate
    # reads this distribution, because a class flood the caps reject is
    # exactly the shift it must see.
    pool_per_class: dict[str, int] = {}
    # Rows the caps dropped, counted instead of vanishing.
    dropped_confident = 0  # confident labels rejected by a cap
    dropped_hard = 0       # hard labels beyond the hard-pool cap
    for rec in records:
        lab = label_record(rec["raw"], router, trust, calibration)
        if lab["tier"] == "confident":
            pool_per_class[lab["schema"]] = (
                pool_per_class.get(lab["schema"], 0) + 1)
            if per_class.get(lab["schema"], 0) >= CAPS["per_class_cycle"]:
                dropped_confident += 1
                continue
            if lab["schema"] == "log_event":
                total_conf = sum(per_class.values()) or 1
                if per_class.get("log_event", 0) / total_conf > CAPS["log_event_share"]:
                    dropped_confident += 1
                    continue
            per_class[lab["schema"]] = per_class.get(lab["schema"], 0) + 1
            confident.append({"id": rec["id"], "source": rec["source"],
                              "raw": rec["raw"], "schema": lab["schema"],
                              "reason": lab["reason"]})
        elif len(hard) < CAPS["hard_cycle"]:
            lab.update({"id": rec["id"], "source": rec["source"]})
            hard.append(lab)
        else:
            dropped_hard += 1
    stats = {"scanned": len(records), "confident": len(confident),
             "hard": len(hard), "per_class": per_class,
             "pool_per_class": pool_per_class,
             "dropped_over_cap": {"confident": dropped_confident,
                                  "hard": dropped_hard},
             "tier1_enabled": _tier1_enabled(calibration),
             "trust": trust_disclosure(measure)}
    return confident, hard, stats
