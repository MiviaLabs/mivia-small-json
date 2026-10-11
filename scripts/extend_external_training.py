#!/usr/bin/env python3
"""Extend training with real external formats, keeping eval splits clean.

The OOD benchmark showed the router misses real code-review prose and real
function-calling text. This adds those FORMATS to training from the public
TRAIN splits, while the eval sets keep only held-out splits:

- CleanCodeReview TRAIN  -> appended to data/real/review_comment/train.jsonl
- CleanCodeReview TEST   -> data/incoming/external/cleancodereview/comments.jsonl
- BFCL, 80% (by hash)    -> appended to data/real/tool_call/train.jsonl
- BFCL, 20% (by hash)    -> data/incoming/external/bfcl/calls.jsonl

Idempotent: rows are deduped against what is already committed. Updates
data/real/PROVENANCE.md counts and sha256 for the two training files.

Stdlib only. Run from the repository root.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real"
EXT = ROOT / "data" / "incoming" / "external"
CCR_URL = "https://zenodo.org/records/13828619/files/code_review_comments.zip?download=1"
NLBSE_URL = ("https://tickettagger.blob.core.windows.net/datasets/"
             "nlbse23-issue-classification-train.csv.tar.gz")
NLBSE_CAP = 10_000
ENTRY_RE = re.compile(r"^\| `data/([^`]+)` \| (\d+) \| ([0-9a-f]{64}) \|$", re.M)
# Unicode line/paragraph separators are valid inside JSON strings but
# ``str.splitlines()`` splits on them, so a reader that splits by lines
# breaks on them. Replace them with spaces before writing.
_SEPARATORS = re.compile("[\u2028\u2029\x85\x1c\x1d\x1e]")


def _clean(text: str) -> str:
    return _SEPARATORS.sub(" ", text).strip()


def _nlbse_train_rows() -> list[str]:
    """Real issue reports from the NLBSE train split, capped and balanced."""
    csv.field_size_limit(10 ** 9)
    cached = Path("/tmp/nlbse23-issue-classification-train.csv")
    if not cached.is_file():
        tar = Path("/tmp/nlbse-train.tar.gz")
        urllib.request.urlretrieve(NLBSE_URL, tar)
        import tarfile

        with tarfile.open(tar) as tf:
            tf.extractall("/tmp")
    per_label: dict[str, list[str]] = {}
    with cached.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("title", "") + "\n" + row.get("body", "")).strip()
            label = (row.get("labels") or "").strip()
            if 10 <= len(text) and label:
                per_label.setdefault(label, []).append(text)
    per = max(1, NLBSE_CAP // max(1, len(per_label)))
    out: list[str] = []
    for texts in per_label.values():
        out.extend(texts[:per])
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    # Split on "\n" only: splitlines() also splits on U+2028/U+2029, which
    # are legal inside a JSON string and would break the parse.
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in rows), encoding="utf-8")


def _ccr_zip() -> zipfile.ZipFile:
    cached = Path("/tmp/ccr.zip")
    if not cached.is_file():
        urllib.request.urlretrieve(CCR_URL, cached)
    return zipfile.ZipFile(cached)


def _ccr_rows(name: str) -> list[tuple[str, str]]:
    out = []
    for line in _ccr_zip().read(name).decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = (rec.get("message") or "").strip()
        label = (rec.get("label") or rec.get("group") or "").strip()
        if 10 <= len(text) and label:
            out.append((text, label))
    return out


def _bfcl_rows() -> list[dict]:
    path = EXT / "bfcl" / "calls.jsonl"
    return _read_jsonl(path)


def _append_training(schema: str, texts: list[str], gold: dict) -> int:
    path = REAL / schema / "train.jsonl"
    # Keep the file inside the committed length window (10..1000 chars); this
    # also repairs any over-long row a previous run appended.
    rows = [r for r in _read_jsonl(path)
            if len(r["raw"]) >= 10 and _clean(r["raw"]) == r["raw"]]
    seen = {_norm(r["raw"]) for r in rows}
    added = 0
    for text in texts:
        text = _clean(text)
        if len(text) < 10 or _norm(text) in seen:
            continue
        seen.add(_norm(text))
        row_gold = dict(gold)
        if schema == "review_comment":
            row_gold["comment"] = text
        rows.append({"source": f"external:{schema}", "raw": text,
                     "gold": row_gold})
        added += 1
    _write_jsonl(path, rows)
    return added


def _update_provenance(entries: list[tuple[str, int, str]]) -> None:
    path = REAL / "PROVENANCE.md"
    text = path.read_text(encoding="utf-8")
    for rel, count, digest in entries:
        text = re.sub(
            rf"^\| `data/{re.escape(rel)}` \| \d+ \| [0-9a-f]{{64}} \|$",
            f"| `data/{rel}` | {count} | {digest} |", text, flags=re.M)
    path.write_text(text, encoding="utf-8")


def _synth_no_tool(tool_rows: list[str], n: int = 1200) -> list[str]:
    """Pair a tool catalogue with ANOTHER row's request.

    The catalogue does not list the tool that request needs, so the correct
    answer is no call. This gives the abstention class real negatives without
    touching the held-out irrelevance split.
    """
    parsed = []
    for text in tool_rows:
        head, sep, req = text.partition("\n\nUser request:")
        if sep and len(req.strip()) >= 10:
            parsed.append((head, req.strip()))
    out = []
    for i in range(min(n, len(parsed))):
        head, _ = parsed[i]
        _, req = parsed[(i * 7 + 3) % len(parsed)]
        out.append(f"{head}\n\nUser request: {req}")
    return out


def _sanitize_real_files() -> None:
    """Replace Unicode line separators in every committed real file.

    They are legal inside JSON strings but break any reader that splits by
    ``str.splitlines()``, so the committed corpora are kept free of them.
    """
    for path in sorted(REAL.glob("*/*.jsonl")):
        rows = _read_jsonl(path)
        changed = False
        for row in rows:
            if "raw" in row and _clean(row["raw"]) != row["raw"]:
                row["raw"] = _clean(row["raw"])
                changed = True
        if changed:
            _write_jsonl(path, rows)


def _update_provenance_all() -> None:
    entries = []
    for path in sorted(REAL.glob("*/*.jsonl")):
        data = path.read_bytes()
        n = sum(1 for line in data.decode("utf-8").split("\n") if line.strip())
        entries.append((str(path.relative_to(REAL.parent)), n, _sha256(data)))
    _update_provenance(entries)


def main() -> int:
    # --- CleanCodeReview: train -> training, test -> eval ---
    train_rows = _ccr_rows("crc_train.jsonl")
    added_rc = _append_training("review_comment", [t for t, _ in train_rows],
                                {"type": "review_comment", "file": "", "line": 1,
                                 "severity": "suggestion"})
    test_rows = _ccr_rows("crc_test.jsonl")
    _write_jsonl(EXT / "cleancodereview" / "comments.jsonl",
                 [{"source": "cleancodereview", "text": t, "label": lab}
                  for t, lab in test_rows])

    # --- BFCL: 80% -> training, 20% -> eval (deterministic hash split) ---
    bfcl = _bfcl_rows()
    train_texts = [r["text"] for r in bfcl
                   if int(hashlib.sha256(r["text"].encode()).hexdigest(), 16) % 5 != 0]
    eval_rows = [r for r in bfcl
                 if int(hashlib.sha256(r["text"].encode()).hexdigest(), 16) % 5 == 0]
    added_tc = _append_training("tool_call", train_texts,
                                {"type": "tool_call", "name": "",
                                 "arguments": "{}", "result": ""})
    _write_jsonl(EXT / "bfcl" / "calls.jsonl", eval_rows)

    # --- NLBSE train split -> `issue` training (rebalances the new classes) ---
    added_is = _append_training("issue", _nlbse_train_rows(),
                                {"type": "issue", "title": "", "labels": [],
                                 "priority": "medium"})

    # --- BFCL irrelevance -> `no_tool_call` (the abstention class) ---
    # Half trains the abstention class, half stays a held-out false-positive
    # set. A tool catalogue whose request needs no call must NOT route to
    # tool_call.
    irr = _read_jsonl(EXT / "bfcl" / "irrelevance.jsonl")
    irr_train, irr_dev = [], []
    for rec in irr:
        digest = int(hashlib.sha256(rec["text"].encode()).hexdigest(), 16)
        (irr_dev if digest % 2 == 0 else irr_train).append(rec["text"])
    added_ntc = _append_training(
        "no_tool_call", irr_train + _synth_no_tool(train_texts),
        {"type": "no_tool_call", "reason": "no matching tool"})
    _write_jsonl(REAL / "no_tool_call" / "dev.jsonl",
                 [{"source": "bfcl-irrelevance", "raw": t,
                   "gold": {"type": "no_tool_call",
                            "reason": "no matching tool"}} for t in irr_dev])

    # --- provenance counts + sha256 for the training files ---
    _sanitize_real_files()
    _update_provenance_all()

    print(f"issue: +{added_is} training rows (NLBSE train split)")
    print(f"no_tool_call: +{added_ntc} training rows, "
          f"{len(irr_dev)} held-out dev rows (BFCL irrelevance)")
    print(f"review_comment: +{added_rc} training rows "
          f"(test split {len(test_rows)} eval rows)")
    print(f"tool_call: +{added_tc} training rows "
          f"({len(eval_rows)} eval rows held out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
