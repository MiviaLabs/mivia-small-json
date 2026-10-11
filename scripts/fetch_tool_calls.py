#!/usr/bin/env python3
"""Fetch real tool-call traces from glaiveai/glaive-function-calling-v2.

Retrieves chat conversations with tool calls directly from the upstream
CloudFront CDN, parses <functioncall> and FUNCTION RESPONSE blocks, and
writes the raw traces to:
data/incoming/tool-calls-glaive.json

Owner decision 2026-10-05: Real public data stored in data/incoming/.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "data" / "incoming" / "tool-calls-glaive.json"
CDN_URL = (
    "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2"
    "/resolve/main/glaive-function-calling-v2.json"
)

FC_PATTERN = re.compile(
    r"<functioncall>\s*(.+?)\s*<\|endoftext\|>"
    r"(?:\s*\n+FUNCTION RESPONSE:\s*(.+?)(?:\n\s*ASSISTANT:|\Z))?",
    re.DOTALL,
)


def extract_calls_from_text(text: str) -> list[dict]:
    # Fix incomplete json at the end of text
    last_obj_end = text.rfind("\n    },")
    if last_obj_end > 0:
        clean_text = text[: last_obj_end + 6] + "\n]"
    elif text.rfind("}") > 0:
        clean_text = text[: text.rfind("}") + 1] + "\n]"
    else:
        clean_text = "[]"

    data = json.loads(clean_text)
    calls = []
    seen = set()
    for item in data:
        chat = item.get("chat", "")
        for m in FC_PATTERN.finditer(chat):
            fc_str = m.group(1).strip()
            resp_str = (m.group(2) or "").strip()
            name_m = re.search(r'"name":\s*"([^"]+)"', fc_str)
            if not name_m:
                continue
            name = name_m.group(1).strip()
            args_m = re.search(r'"arguments":\s*(\'.*?\'|".*?"|\{.*?\})', fc_str, re.DOTALL)
            args = args_m.group(1).strip() if args_m else "{}"
            if args.startswith("'") and args.endswith("'"):
                args = args[1:-1]
            raw = f"<functioncall> {fc_str}\nFUNCTION RESPONSE: {resp_str}".strip()
            if len(raw) > 950:
                resp_cut = resp_str[: max(0, 950 - len(fc_str) - 40)]
                raw = f"<functioncall> {fc_str}\nFUNCTION RESPONSE: {resp_cut}".strip()
            if 10 <= len(raw) <= 1000 and len(name) >= 2 and raw not in seen:
                seen.add(raw)
                calls.append({
                    "source": "glaiveai/glaive-function-calling-v2",
                    "raw": raw,
                    "gold": {
                        "type": "tool_call",
                        "name": name,
                        "arguments": args,
                        "result": resp_str[:400],
                    },
                })
    return calls


def main(target: int = 5500):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching from {CDN_URL}...")
    req = urllib.request.Request(CDN_URL, headers={"User-Agent": "mivia-small-json/1.0"})
    t0 = time.time()

    # Stream chunks of 10MB until we reach the target
    chunks = []
    total_bytes = 0
    all_calls = []

    with urllib.request.urlopen(req, timeout=90) as resp:
        while len(all_calls) < target:
            chunk = resp.read(1024 * 1024 * 8)  # 8MB per read
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)
            combined_text = b"".join(chunks).decode("utf-8", "replace")
            all_calls = extract_calls_from_text(combined_text)
            print(
                f"Downloaded {total_bytes / (1024*1024):.1f} MB, "
                f"extracted {len(all_calls)} / {target} tool calls"
            )

    # Save to OUT_FILE
    OUT_FILE.write_text(json.dumps(all_calls, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(all_calls)} tool calls to {OUT_FILE} "
        f"({OUT_FILE.stat().st_size} bytes in {time.time() - t0:.2f}s)"
    )


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    main(count)
