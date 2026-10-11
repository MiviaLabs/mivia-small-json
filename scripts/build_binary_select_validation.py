"""Build the off-JevBench binary-select validation set (data/eval/).

WHY. The JevBench noul diagnostic (2026-10-10) measured a systematic
yes/no asymmetry in the shipped ``select`` primitive: verbatim-stated
facts answered expected-``no`` (10/35 expected-yes vs 32/39 expected-no).
JevBench is evaluation-only data (PROVENANCE: "never training data"), so
any fix to the binary select mapping MUST be adjudicated on non-JevBench
text first — otherwise the fix is benchmark tuning. This builder produces
that referee: ~500 binary pairs drawn ONLY from the repo's committed real
corpora (data/real/), every label mechanically derived from gold fields
or committed regex rules on the text itself. No JevBench byte is read.

Row semantics (mirrors the shipped binary select call, scripts/
bench_jevbench.py ``answer_row``): the primitive scores the row ``text``
against the binary candidate pair; the row ``predicate`` documents the
question the answer keys on. Two shapes:
- ``question``: the predicate sentence is PREPENDED to the text (the
  caller includes the question — the /v1/select contract allows it);
- ``state``: text only, chosen so the surface carries the polarity
  token (severity/level word) that makes the predicate determinate
  without seeing the question — the JevBench fact-family call shape.

Mechanical label rules (committed here; no vibes, no model in the loop):
- diag-severity: gold ``severity`` field == error (yes) / warning (no).
- diag-tool:     gold ``tool`` field == the tool named in the predicate
                 (yes); a decoy tool name, drawn seeded from the OTHER
                 tools, for the no-side.
- log-level:     gold ``level`` in {ERROR, FATAL} (yes); level == INFO and
                 the raw line matches NO failure-lexicon pattern (no).
- issue-kind:    a gold label contains "bug" and the raw text matches the
                 failure lexicon (yes); a gold label contains "feature
                 request"/"enhancement", the raw matches the request
                 lexicon, and NOT the failure lexicon (no).

Cleaning rules (committed): state-shape rows whose text contains a
standalone "yes"/"no" word are dropped (the select tie-break prefers a
key seen verbatim in the text — such rows would measure the tie-break,
not the ranker); texts are deduped strong-normalized and capped at 1000
characters; per-source quotas balance family x shape x polarity.

Deterministic: seeded random.Random(20261010), stable iteration order,
fixed per-family quotas, and the builder FAILS rather than shipping a
shrunken corpus (governing rule 4). Re-running is byte-identical.

Usage:
    python scripts/build_binary_select_validation.py   # writes data/eval/

Writes data/eval/binary-select-v1.jsonl (+ .sha256) and replaces, never
stacks, the PROVENANCE section in data/eval/binary-select-v1-PROVENANCE.md
(rule 12).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "eval"
OUT_JSONL = OUT_DIR / "binary-select-v1.jsonl"
OUT_SHA = OUT_JSONL.with_suffix(OUT_JSONL.suffix + ".sha256")
OUT_PROV = OUT_DIR / "binary-select-v1-PROVENANCE.md"

SEED = 20261010
MAX_TEXT_CHARS = 1000

DIAG = ROOT / "data" / "real" / "diagnostic" / "train.jsonl"
LOGS = ROOT / "data" / "real" / "log_event" / "train.jsonl"
ISSUES = ROOT / "data" / "real" / "issue" / "train.jsonl"

_SUFFIX = " Answer strictly from the facts stated in the text."
Q_SEVERITY = "Is the severity of this diagnostic error?" + _SUFFIX
Q_LOG = "Does this log entry report an error?" + _SUFFIX
Q_BUG = "Is this a bug report?" + _SUFFIX

# Quotas: (family, shape) -> (n_yes, n_no). Sums to 250/250.
QUOTAS = {
    ("diag-severity", "question"): (48, 48),
    ("diag-severity", "state"): (32, 32),
    ("diag-tool", "question"): (40, 40),
    ("log-level", "question"): (75, 75),
    ("log-level", "state"): (30, 30),
    ("issue-kind", "question"): (25, 25),
}

# Failure lexicon (committed rule; log-level no-side, issue-kind both
# sides). Case-insensitive, word-bounded.
_FAILURE_RE = re.compile(
    r"\b(error|errors|failed|failure|failing|fails|fatal|crash(?:ed|es)?"
    r"|exception|traceback|denied|refused|abort(?:ed|s)?|invalid"
    r"|cannot|could not|unable|not found|missing|broken|breaks|wrong"
    r"|incorrect|regression|does not work|doesn't work|panic|segfault"
    r"|assertion|timed? ?out|unreachable|corrupt)\b",
    re.IGNORECASE)
# Request/feature lexicon (issue-kind no-side) — the text proposes rather
# than reports.
_REQUEST_RE = re.compile(
    r"\b(feature request|enhancement|proposal|propose|it would be "
    r"(?:nice|great|useful)|add support|suggestion|suggest|wish|would "
    r"like|could you add|any chance|request(?:ing)? (?:a )?feature|new "
    r"feature|improvement)\b",
    re.IGNORECASE)
# state-shape cleanliness: no standalone yes/no anywhere in the text
# (the select tie-break prefers a candidate key seen verbatim).
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _qtext(question: str, raw: str) -> str:
    return f"{question}\n{raw}"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


class _Collector:
    """Deduping collector; admits rows with their quota key attached."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.pools: dict[tuple[str, str], dict[str, list[dict]]] = {}

    def admit(self, key: tuple[str, str], source: str, raw: str,
              predicate: str, question: str | None, expected: str) -> None:
        raw = raw.strip()[:MAX_TEXT_CHARS]
        text = _qtext(question, raw) if question is not None else raw
        # Dedupe on the FINAL scored text: the same underlying raw can
        # legitimately appear under two quota keys when its scored text
        # differs (question rows of different predicates, or the state
        # shape), while byte-identical texts may not repeat.
        norm = _norm(text)
        if not norm or norm in self.seen:
            return
        # The standalone yes/no cleanliness rule applies to STATE rows
        # only (the scored text is raw-only there): the select tie-break
        # prefers a candidate key seen verbatim in the text, so such rows
        # would measure the tie-break, not the ranker.
        if key[1] == "state" and _YESNO_RE.search(raw):
            return
        self.seen.add(norm)
        self.pools.setdefault(key, {"yes": [], "no": []})[expected].append(
            {"source": source, "text": text, "predicate": predicate,
             "expected": expected})


def build_candidates() -> dict[tuple[str, str], dict[str, list[dict]]]:
    c = _Collector()

    # -- A. diag-severity (gold severity field) ----------------------------
    for r in load_jsonl(DIAG):
        sev, tool, raw = r.get("severity"), r.get("tool"), r.get("raw") or ""
        if not raw.strip():
            continue
        if sev == "error":
            c.admit(("diag-severity", "question"), f"diagnostic/{tool}",
                    raw, Q_SEVERITY, Q_SEVERITY, "yes")
            c.admit(("diag-severity", "state"), f"diagnostic/{tool}",
                    raw, Q_SEVERITY, None, "yes")
        elif sev == "warning":
            c.admit(("diag-severity", "question"), f"diagnostic/{tool}",
                    raw, Q_SEVERITY, Q_SEVERITY, "no")
            c.admit(("diag-severity", "state"), f"diagnostic/{tool}",
                    raw, Q_SEVERITY, None, "no")

    # -- B. diag-tool (gold tool field; decoy tool for the no-side) --------
    diag_rows = [r for r in load_jsonl(DIAG)
                 if r.get("tool") and (r.get("raw") or "").strip()]
    by_tool: dict[str, list[dict]] = {}
    for r in diag_rows:
        by_tool.setdefault(r["tool"], []).append(r)
    tools = sorted(by_tool)
    rng_tool = random.Random(SEED + 1)
    for tool in tools:
        others = [t for t in tools if t != tool]
        for r in by_tool[tool]:
            raw = r["raw"]
            own = f"Was this diagnostic produced by {tool}?" + _SUFFIX
            c.admit(("diag-tool", "question"), f"diagnostic/{tool}",
                    raw, own, own, "yes")
            decoy = rng_tool.choice(others)
            decoy_q = f"Was this diagnostic produced by {decoy}?" + _SUFFIX
            c.admit(("diag-tool", "question"), f"diagnostic/{tool}",
                    raw, decoy_q, decoy_q, "no")

    # -- C. log-level (gold level + committed failure lexicon) -------------
    for r in load_jsonl(LOGS):
        raw = r.get("raw") or ""
        level = str((r.get("gold") or {}).get("level") or "")
        if not raw.strip():
            continue
        src = str(r.get("source") or "loghub")
        if level in {"ERROR", "FATAL"}:
            c.admit(("log-level", "question"), src, raw, Q_LOG, Q_LOG, "yes")
            c.admit(("log-level", "state"), src, raw, Q_LOG, None, "yes")
        elif level == "INFO" and not _FAILURE_RE.search(raw):
            c.admit(("log-level", "question"), src, raw, Q_LOG, Q_LOG, "no")
            c.admit(("log-level", "state"), src, raw, Q_LOG, None, "no")

    # -- D. issue-kind (gold labels + committed lexicons) ------------------
    for r in load_jsonl(ISSUES):
        raw = r.get("raw") or ""
        labels = " ".join((r.get("gold") or {}).get("labels") or []).lower()
        if not raw.strip():
            continue
        failing = bool(_FAILURE_RE.search(raw))
        if "bug" in labels and failing:
            c.admit(("issue-kind", "question"), "hf:github-issues", raw,
                    Q_BUG, Q_BUG, "yes")
        elif (("feature" in labels or "enhancement" in labels)
              and not failing and _REQUEST_RE.search(raw)):
            c.admit(("issue-kind", "question"), "hf:github-issues", raw,
                    Q_BUG, Q_BUG, "no")

    return c.pools


def assemble() -> list[dict]:
    pools = build_candidates()
    rng = random.Random(SEED)
    rows: list[dict] = []
    shortfalls: list[str] = []
    for (family, shape), (n_yes, n_no) in QUOTAS.items():
        pool = pools.get((family, shape), {"yes": [], "no": []})
        for expected, n in (("yes", n_yes), ("no", n_no)):
            have = pool[expected]
            if len(have) < n:
                shortfalls.append(
                    f"{family}/{shape}/{expected}: have {len(have)}, want {n}")
                n = len(have)
            for r in rng.sample(have, n):
                rows.append({"family": family, "shape": shape, **r})
    if shortfalls:
        raise SystemExit("declared minimum contribution FAILED (rule 4):\n  "
                         + "\n  ".join(shortfalls))
    # Deterministic final order; ids assigned after it.
    rows.sort(key=lambda r: (r["family"], r["shape"], r["expected"],
                             r["source"], r["predicate"]))
    return [{"id": f"bsv-{i:04d}", **r} for i, r in enumerate(rows, 1)]


PROV_TITLE = "# PROVENANCE — data/eval/binary-select-v1.jsonl\n"

PROV_TEMPLATE = """<!-- build:binary-select-v1 (idempotent: the builder OWNS this
     file and rewrites it whole on every run — nothing stacks; rule 12) -->

- **Purpose**: off-JevBench referee for the binary select mapping fix
  (2026-10-10 noul yes/no asymmetry diagnostic). JevBench bytes are NEVER
  read by the builder; this set is evaluation-only for the select surface
  and is not a training source.
- **Builder**: `scripts/build_binary_select_validation.py` (seed
  {seed}, deterministic; re-runs are byte-identical).
- **Sources** (all committed real corpora, license verification deferred
  per governing rule 6; per-source entries live in `data/real/PROVENANCE.md`):
  - `data/real/diagnostic/train.jsonl` (families `diag-severity`,
    `diag-tool`) — gold `severity`/`tool` fields.
  - `data/real/log_event/train.jsonl` (family `log-level`) — LogHub lines,
    gold `level` field + committed failure-lexicon regex.
  - `data/real/issue/train.jsonl` (family `issue-kind`) — GitHub issues,
    gold `labels` + committed failure/request lexicon regexes.
- **Label rules**: mechanical only (see builder docstring) — gold-field
  equality or a committed regex on the row's own text. No model labeled
  anything; no human free-form judgment.
- **Cleaning**: strong-normalized dedupe; state-shape rows containing a
  standalone yes/no dropped (select tie-break cleanliness); text cap
  {max_chars} chars.
- **Counts**: {n_rows} rows — expected-yes {n_yes} / expected-no {n_no};
  by shape: {by_shape}.
- **sha256**: `{sha}`
- **Generated**: {stamp}
"""


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    OUT_JSONL.write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    OUT_SHA.write_text(sha + "  " + OUT_JSONL.name + "\n")

    n_yes = sum(1 for r in rows if r["expected"] == "yes")
    by_shape: dict[str, int] = {}
    for r in rows:
        by_shape[r["shape"]] = by_shape.get(r["shape"], 0) + 1
    by_shape_s = ", ".join(f"{k} {v}" for k, v in sorted(by_shape.items()))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prov = PROV_TITLE + "\n" + PROV_TEMPLATE.format(
        seed=SEED, max_chars=MAX_TEXT_CHARS, n_rows=len(rows), n_yes=n_yes,
        n_no=len(rows) - n_yes, by_shape=by_shape_s, sha=sha, stamp=stamp)
    # The builder owns this file outright: full-file replacement is the
    # idempotency mechanism (re-runs may differ only in the stamp).
    OUT_PROV.write_text(prov)
    print(f"wrote {OUT_JSONL.relative_to(ROOT)} ({len(rows)} rows, "
          f"{n_yes} yes / {len(rows) - n_yes} no)")
    print(f"sha256 {sha}")


def main() -> int:
    write_outputs(assemble())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
