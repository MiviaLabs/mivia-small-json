# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Registration bootstrap for the documented example schema.

``pull_request`` is registered through the public registry API exactly
like any user schema. Every entry point that needs the four-schema
world the committed corpus and artifact were built in — bench, label,
flywheel, synth — calls :func:`register_example_schemas` from here, so
no consumer has to import the synthetic generator. The registration is
idempotent and additive: the built-in tuple never changes.
"""

from __future__ import annotations

import random
from contextlib import contextmanager

from . import registry

PR_TITLES = (
    "Fix login redirect loop on session expiry",
    "Add retry budget to webhook deliveries",
    "Cache computed project quotas for {n} s",
    "Drop duplicated rows from search results page {n}",
    "Split migration runner from the CLI entry point",
    "Handle null avatar in the profile renderer",
    "Export invoices as CSV with unicode columns",
    "Drain poison messages to the dead-letter queue",
    "Cover pagination edge on filtered lists",
    "Bump dependency floor for the security patch",
    "Typo-tolerant query parsing for search v{n}",
    "Rate-limit internal health check endpoints",
)
PR_STATES = ("open", "merged", "closed", "draft")
AUTHORS = ("Jane Doe", "Ravi Patel", "Mia Chen", "Leo Marques", "Sam Ortiz",
           "Nadia Rahman")

PR_SPEC = {
    "type": {"const": "pull_request"},
    "number": {"type": int},
    "title": {"type": str},
    "body": {"type": str, "optional": True},
    "author": {"type": str},
    "state": {"type": str, "enum": PR_STATES},
    "additions": {"type": int},
    "deletions": {"type": int},
}
PR_ROUTING = {
    "strong": (
        r"(?i)\bpull request\b|\bPR #\d+",
        r"(?i)\bmerge[sd]?\b",
        r"\+\d+ -\d+",
        r"(?i)\bpull_request\b",
    ),
    "weak": (r"(?i)\bdiff\b", r"(?i)\bbranch\b", r"(?i)\bdraft\b"),
}
PR_EXTRACTION = {
    "number": (
        r"(?i)\bPR\s*#\s*(\d{1,6})\b",
        r"(?i)\bnumber:\s*(\d{1,6})\b",
        r"#(\d{2,6})\b",
    ),
    "title": (
        r"(?i)\btitle:\s*([^;|\n]+?)\s*(?:::|;|$)",
        r'(?i)"([^"\n]{3,200})"',
        r"(?i)\bPR\s*#\d+:\s*([^\n(+]+?)\s*\(",
    ),
    "author": (
        r"(?i)\bauthor:\s*([^;|\n]+?)\s*(?:::|;|$|\s+state[:=])",
        r"(?i)\bauthor=\s*([^;|\n]+?)\s*(?:::|;|$|\s+state[:=])",
        r"(?i)\bby\s+([A-Z][\w'.-]+(?:\s+[A-Z][\w'.-]+){0,3})",
    ),
    "state": (
        r"(?i)\bstate:\s*(open|merged|closed|draft)\b",
        r"(?i)\bstate=\s*(open|merged|closed|draft)\b",
        r"(?i)\b(merged|draft)\b",
    ),
    "additions": (
        r"(?i)\badditions:\s*(\d{1,6})\b",
        r"\+(\d{1,6})\b",
    ),
    "deletions": (
        r"(?i)\bdeletions:\s*(\d{1,6})\b",
        r"\+\d{1,6}\s*-\s*(\d{1,6})\b",
    ),
}
PR_DEFAULTS = {"number": 0, "title": "", "author": "", "state": "open",
               "additions": 0, "deletions": 0}


def _build_pr_examples(n: int = 40, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({
            "type": "pull_request",
            "number": 100 + i * 13 + rng.randrange(0, 13),
            "title": rng.choice(PR_TITLES).format(n=rng.choice([5, 30, 120])),
            "author": rng.choice(AUTHORS),
            "state": rng.choices(PR_STATES, weights=[35, 30, 25, 10])[0],
            "additions": rng.randrange(1, 900),
            "deletions": rng.randrange(0, 400),
        })
    return out


PR_EXAMPLES = _build_pr_examples()


def register_agent_step_schema() -> None:
    """Register the ``agent_step`` trajectory schema (idempotent).

    Delegates to :mod:`mivia_small_json.agent_step` so the extractor and
    sanitizer wiring lives beside the schema it serves; this entry point
    exists so every consumer that boots the example-schema world
    (bench, label, flywheel, synth, the test suite) sees ``agent_step``
    through the same public registry API.
    """
    from .agent_step import register_agent_step_schema as _register

    _register()


def register_example_schemas() -> None:
    """Register the documented example schemas (idempotent).

    ``pull_request`` plus the multi-turn ``agent_step`` trajectory schema:
    every entry point that needs the corpus world the committed artifact
    was built in — bench, label, flywheel, synth — calls this, so no
    consumer imports the synthetic generator. The registration is
    additive: the built-in tuple never changes.
    """
    if "pull_request" not in registry.active():
        registry.register_schema("pull_request", PR_SPEC, PR_ROUTING,
                                 PR_EXTRACTION, PR_DEFAULTS, PR_EXAMPLES)
    register_agent_step_schema()


def register_text_schema() -> None:
    """The router's trained non-workflow class emits this record (owner
    product decision 2026-10-10): any text is a first-class outcome —
    chat, invoices, news, recipes and other non-developer text
    classifies as plain ``text``. Registers idempotently."""
    if "text" in registry.active():
        return
    registry.register_schema(
        "text",
        spec={
            "type": {"const": "text"},
            "reason": {"type": str},
        },
        routing={"strong": (), "weak": ()},
        extraction={},
        defaults={"reason": "classified as general text"},
    )


def register_all_schemas() -> None:
    """Register every shipped schema (idempotent).

    Long-lived entry points (the sidecar daemon, the MCP server) boot
    this: without it the router's non-built-in predictions fall back to
    rules and the service could only ever emit ``issue``/``log_event``/
    ``commit``. ``tool_clarification`` is included because the
    tool-match guard constructs it on the unfillable-catalogue path.
    """
    from .diagnostic import register_diagnostic_schema
    from .diff_comment import register_diff_and_review_schemas
    from .tool_call import (register_no_tool_call_schema,
                            register_tool_call_schema,
                            register_tool_clarification_schema)

    register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_tool_clarification_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    register_text_schema()


def register_shipped_schema(name: str) -> bool:
    """Register one shipped schema by name (idempotent).

    The routing path calls this on demand when the router predicts a
    class that is not active yet: a shipped schema is always preferable
    to a silent rules fallback that would emit a wrong builtin. Unknown
    names register nothing and return False.
    """
    if name in registry.active():
        return True
    if name == "pull_request":
        registry.register_schema("pull_request", PR_SPEC, PR_ROUTING,
                                 PR_EXTRACTION, PR_DEFAULTS, PR_EXAMPLES)
    else:
        from .agent_step import register_agent_step_schema
        from .diagnostic import register_diagnostic_schema
        from .diff_comment import register_diff_and_review_schemas
        from .tool_call import (register_no_tool_call_schema,
                                register_tool_call_schema,
                                register_tool_clarification_schema)

        registrars = {
            "agent_step": register_agent_step_schema,
            "diagnostic": register_diagnostic_schema,
            "diff": register_diff_and_review_schemas,
            "review_comment": register_diff_and_review_schemas,
            "text": register_text_schema,
            "tool_call": register_tool_call_schema,
            "no_tool_call": register_no_tool_call_schema,
            "tool_clarification": register_tool_clarification_schema,
        }
        registrar = registrars.get(name)
        if registrar is None:
            return False
        registrar()
    return name in registry.active()


@contextmanager
def shipped_world():
    """Enter the full shipped schema world, restoring the prior one on exit.

    ``load_active_router`` boots this globally; measurements and pinned
    fixtures that need it deterministically re-enter it locally.
    """
    with registry.temporary_schemas():
        registry._reset()
        register_all_schemas()
        yield


@contextmanager
def corpus_world():
    """Enter the committed 4+1 corpus world (built-in + example schemas).

    The frozen corpus, trust tables, and spec fingerprint were all
    produced in this world; tests asserting them must not see schemas a
    router boot registered earlier in the process.
    """
    with registry.temporary_schemas():
        registry._reset()
        register_example_schemas()
        yield
