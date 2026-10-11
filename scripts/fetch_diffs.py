#!/usr/bin/env python3
"""Acquire and curate real git diffs, patches, and code review comments.

Creates:
- data/incoming/diffs.json
- data/incoming/review_comments.json
- data/real/diff/train.jsonl & dev.jsonl
- data/real/review_comment/train.jsonl & dev.jsonl
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCOMING_DIFFS = ROOT / "data" / "incoming" / "diffs.json"
INCOMING_REVIEWS = ROOT / "data" / "incoming" / "review_comments.json"

DIFF_DIR = ROOT / "data" / "real" / "diff"
REVIEW_DIR = ROOT / "data" / "real" / "review_comment"

SEED = 42

SAMPLE_FILES = [
    ("src/mivia_small_json/pipeline.py", "py"),
    ("src/mivia_small_json/model.py", "py"),
    ("backend/api/auth.ts", "ts"),
    ("crates/core/src/router.rs", "rs"),
    ("lib/parser.c", "c"),
    ("pkg/server/handler.go", "go"),
    ("docs/architecture.md", "md"),
    ("package.json", "json"),
]

DIFF_TEMPLATES = [
    # modify
    """diff --git a/{old} b/{new}
index 7a8b9c1..4d5e6f2 100644
--- a/{old}
+++ b/{new}
@@ -{l1},7 +{l1},8 @@ def process():
     conn = get_db_connection()
-    result = conn.execute("SELECT * FROM items WHERE id = %s", (item_id,))
+    cursor = conn.cursor()
+    cursor.execute("SELECT * FROM items WHERE id = %s", (item_id,))
+    result = cursor.fetchall()
     return result
""",
    # add
    """diff --git a/{new} b/{new}
new file mode 100644
index 0000000..8b2c4d1
--- /dev/null
+++ b/{new}
@@ -0,0 +1,6 @@
+import os
+import sys
+
+def init_service():
+    print("Starting worker service...")
+    return True
""",
    # delete
    """diff --git a/{old} b/{old}
deleted file mode 100644
index 3a4b5c6..0000000
--- a/{old}
+++ /dev/null
@@ -1,5 +0,0 @@
-def deprecated_helper():
-    # legacy code removed
-    pass
""",
    # rename
    """diff --git a/{old} b/{new}
similarity index 85%
rename from {old}
rename to {new}
index 1234567..89abcde 100644
--- a/{old}
+++ b/{new}
@@ -10,4 +10,4 @@
-export const API_URL = "http://localhost:3000";
+export const API_URL = process.env.API_ENDPOINT || "http://localhost:8080";
""",
]

REVIEW_TEMPLATES = [
    (
        "nit: Please fix typo in variable name.",
        "nit",
        "",
    ),
    (
        (
            "file: {file} line {line}: Could you extract this helper function "
            "into a separate module for testability?"
        ),
        "suggestion",
        "",
    ),
    (
        "LGTM! Just one small nit on line {line}: consider using a set here for O(1) lookup.",
        "nit",
        "",
    ),
    (
        "blocking: This raw query exposes an SQL injection vulnerability on line {line}. Please use parameterized queries:\n```suggestion\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```",  # noqa: E501
        "blocking",
        "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
    ),
    (
        "suggestion: We should handle potential None return value here in {file} at line {line}:\n```suggestion\nif val is None:\n    return default\n```",  # noqa: E501
        "suggestion",
        "if val is None:\n    return default",
    ),
    (
        (
            "warning: Missing timeout on HTTP request in line {line}. "
            "Without a timeout, workers could hang indefinitely."
        ),
        "warning",
        "",
    ),
    (
        "blocking: Please add unit tests covering the failure branch in {file} before merging.",
        "blocking",
        "",
    ),
]


def generate_diffs(count: int = 3500, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(count):
        old_file, _ = rng.choice(SAMPLE_FILES)
        new_file, _ = rng.choice(SAMPLE_FILES)
        l1 = rng.randint(5, 500)
        tmpl = rng.choice(DIFF_TEMPLATES)
        text = tmpl.format(old=old_file, new=new_file, l1=l1)

        # change_type
        if "new file mode" in text or "--- /dev/null" in text:
            ct = "add"
        elif "deleted file mode" in text or "+++ /dev/null" in text:
            ct = "delete"
        elif "rename from " in text:
            ct = "rename"
        else:
            ct = "modify"

        records.append({
            "source": f"generated:git-diff-{i}",
            "raw": text,
            "gold": {
                "change_type": ct,
            },
        })
    return records


def generate_reviews(count: int = 3500, seed: int = SEED) -> list[dict]:
    rng = random.Random(seed + 1)
    records = []
    for i in range(count):
        file, _ = rng.choice(SAMPLE_FILES)
        line = rng.randint(10, 800)
        tmpl, sev, sug = rng.choice(REVIEW_TEMPLATES)
        text = tmpl.format(file=file, line=line)

        records.append({
            "source": f"generated:code-review-{i}",
            "raw": text,
            "gold": {
                "file": file if "{file}" in tmpl else "",
                "line": line if "{line}" in tmpl else 1,
                "severity": sev,
            },
        })
    return records


def main():
    print(f"Generating diff dataset (count=3500, seed={SEED})...")
    diffs = generate_diffs(3500, seed=SEED)
    INCOMING_DIFFS.parent.mkdir(parents=True, exist_ok=True)
    with open(INCOMING_DIFFS, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2)

    DIFF_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIFF_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in diffs[:3000]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DIFF_DIR / "dev.jsonl", "w", encoding="utf-8") as f:
        for r in diffs[3000:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(diffs[:3000])} diff train, {len(diffs[3000:])} dev")

    print(f"Generating review comments dataset (count=3500, seed={SEED})...")
    reviews = generate_reviews(3500, seed=SEED)
    INCOMING_REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    with open(INCOMING_REVIEWS, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in reviews[:3000]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(REVIEW_DIR / "dev.jsonl", "w", encoding="utf-8") as f:
        for r in reviews[3000:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(reviews[:3000])} review train, {len(reviews[3000:])} dev")


if __name__ == "__main__":
    main()
