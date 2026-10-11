#!/usr/bin/env python3
"""Fetch external comparison benchmarks for mivia-small-json.

Stdlib only. These sets are NOT router training data. They are short-text
classification corpora used to compare the small linear model against
published baselines (see code-review-expansion).

Writes:
  data/incoming/external/banking77/{train,test}.csv, categories.json
  data/incoming/external/clinc150/{train,validation,test}.jsonl
  data/incoming/external/PROVENANCE.md

banking77 comes from the upstream PolyAI-LDN repository (its Hugging Face
repo is a loading script, which the datasets-server no longer serves).
clinc_oos comes from the Hugging Face datasets-server rows API (its
parquet files need a parquet reader this project does not carry).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "incoming" / "external"

BANKING_BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/"
                "task-specific-datasets/master/banking_data")
CLINC_ROWS = ("https://datasets-server.huggingface.co/rows"
              "?dataset=clinc%2Fclinc_oos&config=small&split={split}"
              "&offset={offset}&length={length}")
CLINC_INFO = "https://datasets-server.huggingface.co/info?dataset=clinc%2Fclinc_oos"
PAGE = 100
RETRIES = 4


def _get(url: str) -> bytes:
    last = None
    for attempt in range(RETRIES + 3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last = exc
            # 429: the datasets-server throttles aggressively.
            time.sleep((6.0 if exc.code == 429 else 1.5) * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:  # transient
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {RETRIES + 3} tries: {url} ({last})")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_banking77() -> list[str]:
    dest = OUT / "banking77"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in ("train.csv", "test.csv", "categories.json"):
        data = _get(f"{BANKING_BASE}/{name}")
        (dest / name).write_bytes(data)
        rows = max(0, data.count(b"\n") - 1)
        entries.append(
            f"banking77/{name}: {len(data)} B, "
            f"sha256 {_sha256(data)[:16]}, ~{rows} rows"
        )
    return entries


def fetch_clinc150() -> list[str]:
    dest = OUT / "clinc150"
    dest.mkdir(parents=True, exist_ok=True)
    info = json.loads(_get(CLINC_INFO))
    names = (info.get("dataset_info", {}).get("small", {})
             .get("features", {}).get("intent", {}).get("names"))
    if not names:
        raise RuntimeError("clinc_oos: could not read intent label names")

    entries = []
    for split in ("train", "validation", "test"):
        path = dest / f"{split}.jsonl"
        rows: list[dict] = []
        if path.is_file():  # resume a throttled run
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8")
                    .splitlines() if line.strip()]
        offset = len(rows)
        while True:
            page = json.loads(_get(CLINC_ROWS.format(split=split, offset=offset,
                                                     length=PAGE)))
            batch = page.get("rows", [])
            if not batch:
                break
            for item in batch:
                row = item["row"]
                rows.append({"text": row["text"], "intent": names[row["intent"]]})
            offset += len(batch)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in rows), encoding="utf-8")
            total = page.get("num_rows_total", 0)
            if offset >= total:
                break
            time.sleep(0.6)
        entries.append(f"clinc150/{split}.jsonl: {len(rows)} rows")
    (dest / "labels.json").write_text(json.dumps(names, indent=2) + "\n",
                                      encoding="utf-8")
    entries.append(f"clinc150/labels.json: {len(names)} intents")
    return entries


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = fetch_banking77() + fetch_clinc150()
    lines = [
        "# External comparison benchmarks",
        "",
        "Not router training data. Short-text classification sets used to",
        "compare the small linear model against published baselines.",
        "",
        "- banking77: PolyAI-LDN/task-specific-datasets (CC-BY-4.0). 77 intents.",
        "- clinc_oos (small): CLINC150, 50 intents, via Hugging Face datasets-server.",
        "",
        "## Files",
        "",
    ] + [f"- `{e}`" for e in entries] + [""]
    (OUT / "PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")
    for e in entries:
        print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
