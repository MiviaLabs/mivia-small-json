"""Tests for diff and review_comment schemas, routing, and extraction."""

import pytest

from mivia_small_json import pipeline, registry, rules
from mivia_small_json.diff_comment import (
    DIFF_SPEC,
    REVIEW_COMMENT_SPEC,
    extract_diff,
    extract_review_comment,
    register_diff_and_review_schemas,
)
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.validate import validate


@pytest.fixture(autouse=True)
def ensure_schemas_registered():
    register_diff_and_review_schemas()
    yield
    for s in ("diff", "review_comment"):
        if s in registry.active():
            registry.unregister(s)


def test_diff_schema_registration():
    assert "diff" in registry.active()
    assert "review_comment" in registry.active()
    assert registry.resolve("diff") == DIFF_SPEC
    assert registry.resolve("review_comment") == REVIEW_COMMENT_SPEC


def test_diff_rules_routing():
    sample_diff = (
        "diff --git a/src/main.py b/src/main.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -10,4 +10,5 @@\n"
        "-old\n"
        "+new\n"
    )
    assert rules.route(sample_diff) == "diff"


def test_review_comment_rules_routing():
    sample_review = (
        "LGTM with a small nit: please rename variable `foo` to `bar`.\n"
        "```suggestion\n"
        "bar = 42\n"
        "```"
    )
    assert rules.route(sample_review) == "review_comment"

    blocking_review = "blocking: SQL injection vulnerability detected on line 15."
    assert rules.route(blocking_review) == "review_comment"


def test_diff_extraction():
    diff_text = (
        "diff --git a/src/old.py b/src/new.py\n"
        "similarity index 90%\n"
        "rename from src/old.py\n"
        "rename to src/new.py\n"
        "--- a/src/old.py\n"
        "+++ b/src/new.py\n"
        "@@ -1,3 +1,4 @@\n"
        " line 1\n"
        "-line 2\n"
        "+line 2 modified\n"
        "+line 3 added\n"
    )
    extracted = extract_diff(diff_text)
    assert extracted["type"] == "diff"
    assert extracted["old_path"] == "src/old.py"
    assert extracted["new_path"] == "src/new.py"
    assert extracted["change_type"] == "rename"
    assert extracted["additions"] == 2
    assert extracted["deletions"] == 1
    assert validate(extracted, "diff") == []


def test_review_comment_extraction():
    review_text = (
        "blocking: this raw query in file: `src/api.py` line 42 needs sanitization:\n"
        "```suggestion\n"
        "cursor.execute('SELECT * FROM users WHERE id = %s', (uid,))\n"
        "```"
    )
    extracted = extract_review_comment(review_text)
    assert extracted["type"] == "review_comment"
    assert extracted["file"] == "src/api.py"
    assert extracted["line"] == 42
    assert extracted["severity"] == "blocking"
    assert "cursor.execute" in extracted["suggested_code"]
    assert validate(extracted, "review_comment") == []


def test_diff_and_review_routing_model():
    router = pipeline.load_router(MODEL_PATH)
    assert router is not None

    diff_sample = (
        "diff --git a/src/main.py b/src/main.py\n"
        "index 1234567..89abcde 100644\n"
        "--- a/src/main.py\n"
        "+++ b/src/main.py\n"
        "@@ -10,4 +10,5 @@\n"
        "-old\n"
        "+new\n"
    )
    review_sample = "LGTM with a small nit: please rename variable `foo` to `bar`."

    assert router.predict([diff_sample])[0] == "diff"
    assert router.predict([review_sample])[0] == "review_comment"


def test_review_line_requires_the_word_line():
    """A bare L<digits> (e.g. "L2 cache") must not read as a line number."""
    extracted = extract_review_comment("nit: the L2 cache lookup is slow here")
    assert extracted["line"] == 1, extracted["line"]


def test_pipeline_classify_extracts_diff_counts():
    """Pipeline classify for diff schema uses extract_diff and counts changes."""
    patch = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,1 +1,2 @@\n"
        "-old\n"
        "+new\n"
        "+added\n"
    )
    obj, meta = pipeline.classify(patch)
    assert meta["valid"] is True
    assert obj["type"] == "diff"
    assert obj["old_path"] == "foo.py"
    assert obj["new_path"] == "foo.py"
    assert obj["additions"] == 2
    assert obj["deletions"] == 1


