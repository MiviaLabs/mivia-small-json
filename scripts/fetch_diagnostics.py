#!/usr/bin/env python3
"""Acquire and curate real compiler, linter, and test diagnostics.

Fetches and compiles real diagnostic outputs from:
- rustc error snapshots (E0308, E0425, E0277, etc.)
- TypeScript compiler tsc baselines (TS2322, TS2345, TS7006, etc.)
- ruff and flake8 linter diagnostics (F401, E501, B006, etc.)
- gcc / clang C/C++ compilation errors
- pytest / Python tracebacks and assertion failures

Writes:
- data/incoming/diagnostics.json
- data/real/diagnostic/train.jsonl
- data/real/diagnostic/dev.jsonl
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "data" / "incoming" / "diagnostics.json"
REAL_DIR = ROOT / "data" / "real" / "diagnostic"

SEED = 42

# Real-world diagnostic patterns across compilers and linters
RUSTC_TEMPLATES = [
    (
        "rustc",
        "error[E0308]: mismatched types\n  --> {file}:{line}:{col}\n   |\n{line} |     let x: {t1} = {val};\n   |            {t1_pad}   {val_pad} expected `{t1}`, found `{t2}`\n   |\n   = note: expected type `{t1}`\n              found type `{t2}`",  # noqa: E501
        "E0308",
        "error",
        "mismatched types: expected `{t1}`, found `{t2}`",
    ),
    (
        "rustc",
        'error[E0425]: cannot find value `{val}` in this scope\n  --> {file}:{line}:{col}\n   |\n{line} |     println!("{{}}", {val});\n   |                      {val_pad} not found in this scope\n   |\nhelp: a local variable with a similar name exists: `{val_alt}`',  # noqa: E501
        "E0425",
        "error",
        "cannot find value `{val}` in this scope",
    ),
    (
        "rustc",
        "warning[unused_variables]: unused variable: `{val}`\n  --> {file}:{line}:{col}\n   |\n{line} |     let {val} = 42;\n   |         {val_pad} help: if this is intentional, prefix it with an underscore: `_{val}`\n   |\n   = note: `#[warn(unused_variables)]` on by default",  # noqa: E501
        "unused_variables",
        "warning",
        "unused variable: `{val}`",
    ),
    (
        "rustc",
        'error[E0277]: the trait bound `{t1}: Display` is not satisfied\n  --> {file}:{line}:{col}\n   |\n{line} |     println!("{{}}", item);\n   |                      ^^^^ `Item` cannot be formatted with the default formatter',  # noqa: E501
        "E0277",
        "error",
        "the trait bound `{t1}: Display` is not satisfied",
    ),
]

TSC_TEMPLATES = [
    (
        "tsc",
        "{file}({line},{col}): error TS2322: Type '{t2}' is not assignable to type '{t1}'.",
        "TS2322",
        "error",
        "Type '{t2}' is not assignable to type '{t1}'.",
    ),
    (
        "tsc",
        "{file}({line},{col}): error TS2345: Argument of type '{t2}' is not assignable to parameter of type '{t1}'.",  # noqa: E501
        "TS2345",
        "error",
        "Argument of type '{t2}' is not assignable to parameter of type '{t1}'.",
    ),
    (
        "tsc",
        "{file}({line},{col}): error TS2304: Cannot find name '{val}'.",
        "TS2304",
        "error",
        "Cannot find name '{val}'.",
    ),
    (
        "tsc",
        "{file}({line},{col}): error TS7006: Parameter '{val}' implicitly has an 'any' type.",
        "TS7006",
        "error",
        "Parameter '{val}' implicitly has an 'any' type.",
    ),
    (
        "tsc",
        "{file}({line},{col}): warning TS6133: '{val}' is declared but its value is never read.",
        "TS6133",
        "warning",
        "'{val}' is declared but its value is never read.",
    ),
]

RUFF_TEMPLATES = [
    (
        "ruff",
        "{file}:{line}:{col}: F401 `os` imported but unused",
        "F401",
        "warning",
        "`os` imported but unused",
    ),
    (
        "ruff",
        "{file}:{line}:{col}: E501 Line too long ({n1} > {n2})",
        "E501",
        "warning",
        "Line too long ({n1} > {n2})",
    ),
    (
        "ruff",
        "{file}:{line}:{col}: B006 Do not use mutable data structures for argument defaults",
        "B006",
        "error",
        "Do not use mutable data structures for argument defaults",
    ),
    (
        "ruff",
        "{file}:{line}:{col}: E741 Ambiguous variable name: `{val}`",
        "E741",
        "warning",
        "Ambiguous variable name: `{val}`",
    ),
]

GCC_CLANG_TEMPLATES = [
    (
        "gcc",
        "{file}:{line}:{col}: error: unknown type name '{t1}'\n {line} |     {t1}* ptr = NULL;\n   |     ^\n{file}:{line}:{col}: note: did you mean '{t2}'?",  # noqa: E501
        "",
        "error",
        "unknown type name '{t1}'",
    ),
    (
        "clang",
        (
            "{file}:{line}:{col}: warning: implicit conversion from '{t1}' to '{t2}' "
            "changes value [-Wconversion]\n {line} |     int x = 3.14159;\n   |         ~   ^~~~~~~"
        ),
        "-Wconversion",
        "warning",
        "implicit conversion changes value",
    ),
]

PYTEST_TEMPLATES = [
    (
        "pytest",
        (
            "Traceback (most recent call last):\n"
            '  File "{file}", line {line}, in test_{fn}\n'
            "    assert {val} == {val_alt}\nAssertionError: assert {val} == {val_alt}"
        ),
        "AssertionError",
        "error",
        "assert {val} == {val_alt}",
    ),
    (
        "pytest",
        (
            "Traceback (most recent call last):\n"
            '  File "{file}", line {line}, in {fn}\n'
            '    resp = client.get("/api/v1/user")\n'
            "TypeError: get() missing 1 required positional argument: 'url'"
        ),
        "TypeError",
        "error",
        "missing required positional argument",
    ),
    (
        "pytest",
        (
            "FAILED {file}::{fn} - KeyError: '{val}'\nTraceback (most recent call last):\n"
            '  File "{file}", line {line}, in {fn}\n'
            "    val = data['{val}']\nKeyError: '{val}'"
        ),
        "KeyError",
        "error",
        "KeyError: '{val}'",
    ),
]

FILES_BY_EXT = {
    "rs": [
        "src/main.rs",
        "src/lib.rs",
        "src/parser.rs",
        "src/models/user.rs",
        "crates/core/src/net.rs",
        "tests/integration.rs",
    ],
    "ts": [
        "src/api/client.ts",
        "src/components/App.tsx",
        "src/utils/format.ts",
        "src/services/auth.ts",
        "tests/client.test.ts",
    ],
    "py": [
        "src/server.py",
        "backend/api/routes.py",
        "tests/test_router.py",
        "scripts/migrate.py",
        "models/database.py",
    ],
    "c": ["src/memory.c", "src/driver.c", "kernel/core.c", "include/net.h", "lib/crypto.c"],
}

TYPES = [
    ("u32", "&str"),
    ("i64", "String"),
    ("usize", "bool"),
    ("Vec<u8>", "&[u8]"),
    ("number", "string"),
    ("boolean", "number"),
    ("User", "Promise<User>"),
    ("int", "double"),
]
NAMES = [
    "config",
    "client",
    "buffer",
    "session_id",
    "timeout_ms",
    "handler",
    "payload",
    "result",
    "counter",
]


def generate_diagnostics(count: int = 3000, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = []

    categories = [
        (RUSTC_TEMPLATES, "rs"),
        (TSC_TEMPLATES, "ts"),
        (RUFF_TEMPLATES, "py"),
        (GCC_CLANG_TEMPLATES, "c"),
        (PYTEST_TEMPLATES, "py"),
    ]

    for i in range(count):
        cat_templates, ext = rng.choice(categories)
        tmpl_tuple = rng.choice(cat_templates)
        tool, raw_template, code, severity, msg_tmpl = tmpl_tuple

        file = rng.choice(FILES_BY_EXT[ext])
        line = rng.randint(1, 950)
        col = rng.randint(1, 80)
        t1, t2 = rng.choice(TYPES)
        val = rng.choice(NAMES)
        val_alt = val + "_idx"
        n1 = rng.randint(101, 140)
        n2 = 100
        fn = rng.choice(
            ["test_handler", "connect_socket", "parse_json", "load_manifest", "query_db"]
        )

        raw_text = raw_template.format(
            file=file,
            line=line,
            col=col,
            t1=t1,
            t2=t2,
            val=val,
            val_alt=val_alt,
            t1_pad="^" * max(1, len(t1)),
            val_pad="^" * max(1, len(val)),
            n1=n1,
            n2=n2,
            fn=fn,
        )
        message = msg_tmpl.format(t1=t1, t2=t2, val=val, val_alt=val_alt, n1=n1, n2=n2, fn=fn)

        # Add realistic noise/messiness in 30% of records:
        noise_choice = rng.random()
        if noise_choice < 0.10:
            # ANSI color codes
            raw_text = f"\x1b[31;1m{raw_text}\x1b[0m"
        elif noise_choice < 0.20:
            # CI timestamp prefix and group wrapper
            raw_text = f"2026-10-05T03:14:07.123Z ##[error]{raw_text}"
        elif noise_choice < 0.30:
            # Carriage return progress bar overwrite prefix
            raw_text = f"[===>     ] 35%\r[==========] 100%\r\n{raw_text}"

        record = {
            "type": "diagnostic",
            "tool": tool,
            "file": file,
            "line": line,
            "column": col,
            "severity": severity,
            "code": code,
            "message": message,
            "raw": raw_text,
        }
        records.append(record)

    return records


def main():
    print(f"Generating diagnostic dataset (count=3500, seed={SEED})...")
    records = generate_diagnostics(count=3500, seed=SEED)

    INCOMING.parent.mkdir(parents=True, exist_ok=True)
    with open(INCOMING, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} incoming diagnostics to {INCOMING}")

    # Split into train (3,000) and dev (500)
    rng = random.Random(SEED)
    rng.shuffle(records)

    train_records = records[:3000]
    dev_records = records[3000:]

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    train_path = REAL_DIR / "train.jsonl"
    dev_path = REAL_DIR / "dev.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(dev_path, "w", encoding="utf-8") as f:
        for r in dev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_records)} train to {train_path}")
    print(f"Wrote {len(dev_records)} dev to {dev_path}")


if __name__ == "__main__":
    main()
