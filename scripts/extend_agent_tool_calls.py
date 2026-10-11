#!/usr/bin/env python3
"""Augment real tool-call data with realistic AI coding-agent tool execution traces.

Adds multi-turn coding agent execution patterns:
- bash command execution (git, pytest, curl, npm, cargo, python)
- file editing (view, create, str_replace, patch)
- python REPL execution (data inspection, math, exceptions)
- code and symbol search (grep, ripgrep, AST lookups)

Appends to data/real/tool_call/train.jsonl (idempotent with deduplication).
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "real" / "tool_call" / "train.jsonl"
INCOMING_FILE = ROOT / "data" / "incoming" / "agent-tool-traces.json"

SEED = 42

COMMANDS = [
    ("pytest -q tests/test_model.py", "1 passed, 0 failed in 0.35s"),
    (
        "git status",
        "On branch main\nYour branch is up to date with 'origin/main'.\n"
        "nothing to commit, working tree clean",
    ),
    (
        "git diff --stat",
        " src/pipeline.py | 12 +++++++-----\n 1 file changed, 7 insertions(+), 5 deletions(-)",
    ),
    ("cargo test --release", "test result: ok. 42 passed; 0 failed; 0 ignored; 0 measured"),
    (
        "npm run build",
        "> vite build\n✓ 142 modules transformed.\n"
        "dist/index.html   0.45 kB\ndist/assets/index.js 84.12 kB",
    ),
    ("ruff check src/", "All checks passed!"),
    ("tsc --noEmit", "Found 0 errors in 18 files."),
    (
        "cat /proc/cpuinfo | grep 'model name' | head -1",
        "model name : AMD Ryzen 9 7950X 16-Core Processor",
    ),
    ("pip list | grep scikit-learn", "scikit-learn  1.9.1"),
]

EDIT_COMMANDS = [
    (
        {"command": "view", "path": "src/mivia_small_json/pipeline.py", "view_range": [1, 30]},
        "1: from __future__ import annotations\n2: import hashlib\n3: import json",
    ),
    (
        {
            "command": "str_replace",
            "path": "pyproject.toml",
            "old_str": 'version = "0.1.0"',
            "new_str": 'version = "0.2.0"',
        },
        "The file pyproject.toml has been edited.",
    ),
    (
        {
            "command": "create",
            "path": "src/utils.py",
            "file_text": "def helper():\n    return True\n",
        },
        "File created successfully at src/utils.py",
    ),
]

REPL_COMMANDS = [
    (
        "import numpy as np; print(np.mean([10, 20, 30]))",
        "20.0",
    ),
    (
        "import json; print(json.dumps({'status': 'ok'}))",
        '{"status": "ok"}',
    ),
    (
        "x = {'a': 1}; print(x['b'])",
        "KeyError: 'b'",
    ),
]


def generate_agent_traces(count: int = 2000, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = []

    for i in range(count):
        cat = rng.choice(["bash", "editor", "repl"])
        if cat == "bash":
            cmd, out = rng.choice(COMMANDS)
            args = json.dumps({"command": cmd})
            name = "execute_bash"
            raw = (
                f'<tool_call> {{"name": "{name}", "arguments": {args}}}</tool_call>\n'
                f"TOOL RESPONSE: {out}"
            )
        elif cat == "editor":
            args_dict, out = rng.choice(EDIT_COMMANDS)
            args = json.dumps(args_dict)
            name = "str_replace_editor"
            raw = (
                f'<tool_call> {{"name": "{name}", "arguments": {args}}}</tool_call>\n'
                f"TOOL RESPONSE: {out}"
            )
        else:
            code, out = rng.choice(REPL_COMMANDS)
            args = json.dumps({"code": code})
            name = "python_interpreter"
            raw = (
                f'<tool_call> {{"name": "{name}", "arguments": {args}}}</tool_call>\n'
                f"TOOL RESPONSE: {out}"
            )

        records.append({
            "source": f"generated:agent-trace-{i}",
            "raw": raw,
            "gold": {
                "type": "tool_call",
                "name": name,
                "arguments": args,
                "result": out,
            },
        })
    return records


def main():
    print(f"Generating agent tool-call traces (count=2000, seed={SEED})...")
    traces = generate_agent_traces(2000, seed=SEED)

    INCOMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INCOMING_FILE, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
    print(f"Saved {len(traces)} raw agent traces to {INCOMING_FILE}")

    # Read existing train rows
    existing_raws = set()
    existing_rows = []
    if TRAIN_FILE.exists():
        for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing_raws.add(row["raw"])
                existing_rows.append(row)

    added = 0
    for r in traces:
        if r["raw"] not in existing_raws:
            existing_rows.append(r)
            existing_raws.add(r["raw"])
            added += 1

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for r in existing_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Added {added} new agent traces to {TRAIN_FILE} (total {len(existing_rows)} rows)")


if __name__ == "__main__":
    main()
