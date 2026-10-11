"""Benchmark for Zero-Shot Primitives: select(), verify(), and grade().

Empirically measures:
1. Primitive 1: select (Multi-candidate capability selection and prompt routing)
2. Primitive 2: verify (Binary semantic entailment on 1,706 held-out BFCL pairs)
3. Primitive 3: grade (Ordinal triage across complexity and severity levels)
4. CPU execution latency (p50 / p95) and peak RSS memory.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import resource
import time
from pathlib import Path

import numpy as np

from mivia_small_json import tool_match

ROOT = Path(__file__).resolve().parents[1]

# 1. Dataset for prompt-to-model routing (select)
PROMPT_ROUTING_MODELS = {
    "math_model": (
        "math.solve: Solve complex mathematical equations, calculate roots of quadratic formulas, "
        "calculus, and algebra."
    ),
    "code_model": (
        "code.generate: Write Python code, implement algorithms, debug functions, "
        "and software programs."
    ),
    "sql_model": (
        "sql.query: Run SQL queries, database joins, schema mutations, and data aggregations."
    ),
    "sysadmin_model": (
        "system.manage: Linux bash commands, systemctl services, restart docker containers, "
        "and server infrastructure."
    ),
}

PROMPT_ROUTING_DATASET = [
    ("Calculate the roots of a quadratic equation where a=2, b=5, and c=3", "math_model"),
    ("Find the derivative of f(x) = 3x^3 - 5x + 2 with respect to x", "math_model"),
    ("Compute the factorial of 12 using mathematical functions", "math_model"),
    ("Write a Python function to compute the Fibonacci sequence using memoization", "code_model"),
    ("Implement a binary search algorithm in Python with O(log n) complexity", "code_model"),
    ("Debug this Python IndexError in list comprehension parsing", "code_model"),
    ("Run a SQL query to select department and average salary from employees table", "sql_model"),
    ("SELECT u.id, count(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id", "sql_model"),
    ("Perform an ALTER TABLE migration to add an index on created_at timestamp", "sql_model"),
    ("Restart the docker container and check nginx service logs", "sysadmin_model"),
    ("Inspect systemd journalctl unit status for postgresql service failure", "sysadmin_model"),
    ("Check disk space utilization on root partition with df -h command", "sysadmin_model"),
]

# 1b. Ten-domain adversarial prompt-to-model routing. Each domain contributes
# four conditions: clean (explicit ask), terse (single word or short fragment),
# messy (stack trace, preamble, or markdown wrapper), boundary (mentions
# another domain's vocabulary while asking for this one).
ROUTING_DOMAINS_10 = {
    "math": (
        "math.solve: Solve mathematical equations and numeric computation: "
        "quadratic and polynomial roots, derivatives, integrals, matrix "
        "algebra, and arithmetic."
    ),
    "python": (
        "python.dev: Write, debug, and review Python code: functions, "
        "scripts, algorithms, libraries, and error fixes."
    ),
    "rust": (
        "rust.dev: Write and debug Rust programs: cargo builds, the borrow "
        "checker, lifetimes, and systems code."
    ),
    "sql": (
        "sql.query: Write and run SQL for relational databases: SELECT "
        "queries, joins, indexes, migrations, and stored procedures."
    ),
    "devops": (
        "devops.run: Operate servers and infrastructure: restart services, "
        "systemd and nginx, docker containers, kubernetes deployments, and "
        "checking logs and disk space."
    ),
    "git": (
        "git.ops: Version control with git: commits, branches, rebases, "
        "merges, stash, pull requests, and undoing changes."
    ),
    "docs": (
        "docs.write: Write technical documentation: README files, API "
        "references, setup guides, and usage examples."
    ),
    "security": (
        "security.audit: Find and fix security issues: vulnerabilities, "
        "injection, hardcoded secrets, CVEs, and unsafe code."
    ),
    "translate": (
        "translate.lang: Translate text between human languages, preserving "
        "names and technical terms."
    ),
    "creative": (
        "creative.write: Write creative fiction: stories, poems, scripts, "
        "and imaginative prose."
    ),
}

ROUTING_DATASET_10 = [
    # math
    ("Calculate the integral of 3x^2 from 0 to 2", "math", "clean"),
    ("derivative", "math", "terse"),
    ("hey can you help me, my calculator broke:\nTraceback (most recent call last):\n"
     "  File \"calc.py\", line 12, in <module>\nNameError: name 'integral' is not defined\n"
     "I need to compute the area under the curve of f(x) = 5x^2 between 1 and 4", "math", "messy"),
    ("I have a Python list of quadratic expressions; take the integral of each polynomial "
     "and print the total area as a single number", "math", "boundary"),
    # python
    ("Write a Python function that deduplicates a list while preserving order", "python", "clean"),
    ("python", "python", "terse"),
    ("```\nimport os\n\ndef f(p):\n    return os.listdir(p)\n```\n"
     "make this python function also skip hidden files", "python", "messy"),
    ("My Rust crate needs a Python FFI helper: write the Python side that loads the "
     "shared library and calls into it", "python", "boundary"),
    # rust
    ("Fix this Rust borrow checker error: cannot borrow `v` as mutable more than once "
     "in a cargo project", "rust", "clean"),
    ("rust", "rust", "terse"),
    ("$ cargo build --release\nerror[E0507]: cannot move out of `ctx.config` which is "
     "behind a shared reference\nhelp: consider borrowing it; the function should "
     "take ownership and clone first", "rust", "messy"),
    ("In my Python service I want a Rust extension module to speed up the parser loop",
     "rust", "boundary"),
    # sql
    ("Write a query that returns the top 5 customers by total order value this year",
     "sql", "clean"),
    ("sql", "sql", "terse"),
    ("From: analytics team\nSubject: dashboard broken\n\nthe monthly report query "
     "SELECT month, sum(revenue) FROM orders GROUP BY month stopped working after the "
     "schema change; rewrite it for the new orders_v2 table", "sql", "messy"),
    ("Write a Postgres function that logs every INSERT into the users table to an "
     "audit trail table", "sql", "boundary"),
    # devops
    ("Restart the nginx service on the staging host and verify it came back up", "devops", "clean"),
    ("devops", "devops", "terse"),
    ("---\nrainbow of logs:\n Sep 12 03:11:01 web-01 systemd[1]: nginx.service: Failed "
     "with result 'exit-code'.\n---\nthe container keeps crash-looping, roll the "
     "deployment back and free disk space on the node", "devops", "messy"),
    ("Write a git pre-commit hook that runs kubectl config validation before every "
     "deployment commit", "git", "boundary"),
    # git
    ("Undo the last commit but keep its changes staged in the working tree", "git", "clean"),
    ("git", "git", "terse"),
    ("Ugh I rebased onto the wrong branch and now main has my WIP commits:\n"
     "  git log --oneline\n  a1b2c3 (HEAD -> main) wip: experiment\n"
     "  d4e5f6 wip: more\nmove them back to a feature branch", "git", "messy"),
    ("Write a Python script that parses `git log` output and opens a PR for each "
     "merged release branch", "python", "boundary"),
    # docs
    ("Write a README section documenting the config file format and all options", "docs", "clean"),
    ("docs", "docs", "terse"),
    ("```markdown\n## API\n\nPOST /v1/items\n```\nexpand this stub into full API "
     "reference documentation with request and response examples", "docs", "messy"),
    ("Our security audit report needs the remediation steps rewritten as user-facing "
     "documentation", "docs", "boundary"),
    # security
    ("Scan this Express route for SQL injection and secrets leakage vulnerabilities",
     "security", "clean"),
    ("security", "security", "terse"),
    ("[!] alert from the scanner:\n  CVE-2025-1234 reference found in package-lock.json\n"
     "  hardcoded AWS key in settings.py\ntriage these findings and check the rest of "
     "the repo for similar credential leaks", "security", "messy"),
    ("Write a creative backstory for our security mascot to make the awareness "
     "training less boring", "creative", "boundary"),
    # translate
    ("Translate this error message into Japanese, keeping the exception class names "
     "in English", "translate", "clean"),
    ("translate", "translate", "terse"),
    ("User report (de): \"Beim Speichern der Datei ist ein Fehler aufgetreten.\"\n"
     "I need the German and French versions of every UI string in this table, "
     "translated", "translate", "messy"),
    ("Translate the SQL tutorial chapter from English to Spanish for our docs site",
     "translate", "boundary"),
    # creative
    ("Write a short story about a lighthouse keeper who finds a message in a bottle",
     "creative", "clean"),
    ("story", "creative", "terse"),
    ("Hi! So for the team offsite we need something fun to open the night, like a "
     "poem or a funny short story about our startup running out of coffee", "creative", "messy"),
    ("Write a detective story where a grumpy sysadmin hunts down a mysterious "
     "memory leak through the night", "creative", "boundary"),
]

# 2. Dataset for grade (Ordinal complexity anchors)
# NOTE: These are zero-shot grading anchors — ordered complexity tiers the
# grade() primitive ranks against — deliberately NOT the schema's severity
# vocabulary `schemas.LEVELS` ("TRACE".."FATAL"), which is the output enum
# for the `level` field of log_event/diagnostic records.
GRADE_ANCHORS = [
    "level_1_basic: Simple arithmetic addition and subtraction operations.",
    "level_2_intermediate: Polynomial equations, quadratic formulas, and algebra.",
    "level_3_advanced: Multivariable calculus, differential equations, and linear algebra.",
]

GRADE_DATASET = [
    ("What is 45 plus 87?", "level_1_basic"),
    ("Add 12 and 38 together", "level_1_basic"),
    ("Find the roots of the quadratic equation 2x^2 + 7x + 3 = 0", "level_2_intermediate"),
    ("Solve the system of linear equations 3x + 2y = 8 and x - y = 1", "level_2_intermediate"),
    ("Evaluate double integral of xy over region bounded by x=0, y=x^2", "level_3_advanced"),
    ("Find the eigenvalues and eigenvectors of a 3x3 symmetric matrix", "level_3_advanced"),
]

# 2b. Severity grade benchmark on real held-out data: LogHub dev rows
# (data/real/log_event/dev.jsonl). The guard corpus is BFCL-only, so every
# LogHub row is disjoint from everything the cross-encoders trained on.
# Rows are split by whether the gold level token appears verbatim in the
# text: implicit rows (primary; no lexical shortcut) vs explicit (secondary;
# partly lexical). FATAL folds into the CRITICAL tier as a label alias.
SEVERITY_TIERS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
SEVERITY_LEVEL_ALIAS = {"FATAL": "CRITICAL", "WARN": "WARNING", "ERR": "ERROR"}
SEVERITY_ANCHORS_EXCLUSIVE = [
    "info level: routine informational messages about normal successful "
    "operation and expected events.",
    "warning level: recoverable abnormal conditions that may need attention "
    "but are not failures.",
    "error level: operation failures, faults, and errors that disrupted a "
    "task or transaction.",
    "critical level: severe crashes, fatal errors, data loss, or complete "
    "system failure requiring immediate action.",
]
SEVERITY_ANCHORS_CUMULATIVE = [
    "info level: routine informational messages about normal successful "
    "operation and expected events.",
    "info level or above: routine operation plus recoverable abnormal "
    "conditions that may need attention.",
    "warning level or above: recoverable conditions plus operation failures "
    "and faults that disrupted a task.",
    "error level or above: failures plus severe crashes, fatal errors, data "
    "loss, or complete system failure requiring immediate action.",
]
_LEVEL_TOKEN = None

# 2c. Novel-register severity grade set: PRE-REGISTERED eval, frozen before
# the register-expansion training run (grade-data-expansion).
# 10 hand-written business/SRE-register lines per tier, no level tokens, no
# LogHub phrasing, never used as training rows. Bar: >= 80% exact.
NOVEL_REGISTER_TIERS = ["INFO", "WARNING", "ERROR", "CRITICAL"]
NOVEL_REGISTER_GRADE_CASES = [
    # INFO: normal operation, in business/product language.
    ("Maria from accounts confirmed the Q3 reconciliation report exported cleanly", 0),
    ("The nightly job synced 4,318 customer records to the CRM with no rejections", 0),
    ("All 12 checkout pods report healthy after this morning's deploy", 0),
    ("The new pricing page went live at 09:00 and conversions are tracking normally", 0),
    ("Customer support dashboards show median response time back under two minutes", 0),
    ("User alice logged in from 10.0.4.2 and opened the billing dashboard", 0),
    ("The quarterly board-deck export completed and was shared with the leadership channel", 0),
    ("This morning's invoice batch printed 2,904 invoices and archived them to cold storage", 0),
    ("Both office printers are back online after the firmware update", 0),
    ("The design team's new icon set shipped to the component library without conflicts", 0),
    # WARNING: degradation, risk, needs attention but still serving.
    ("European customers are seeing the recommendations widget load about twice as "
     "slow as usual since yesterday's release", 1),
    ("The support inbox is backing up, oldest unanswered ticket is now six hours old", 1),
    ("Free tier signups exceeded forecast by forty percent and the onboarding "
     "database is at 82 percent of its storage allowance", 1),
    ("The payment provider warned us they are decommissioning the old API endpoint "
     "we still call for refunds", 1),
    ("Two of the five warehouse scanners keep dropping their wifi connection and "
     "need manual reconnects each shift", 1),
    ("Cache miss rate is climbing, the upstream looks slow but we are still serving", 1),
    ("Payment provider latency above four seconds, requests are being retried", 1),
    ("The staging environment has been out of disk headroom all afternoon and "
     "deploys to it are queueing", 1),
    ("Legal flagged that the retention policy we promised enterprise customers is "
     "not what the archive job actually does", 1),
    ("The beta feature flag is on for more users than the rollout plan intended", 1),
    # ERROR: operations failing, tasks disrupted.
    ("Invoice generation started failing for all customers on the annual plan "
     "shortly after the tax-table update", 2),
    ("The nightly inventory sync to the ERP has aborted three nights in a row, "
     "counts are drifting from the warehouse system", 2),
    ("Customers using Safari cannot complete password reset, the email link "
     "throws a certificate error", 2),
    ("The reporting endpoint returns a 500 for any date range wider than one "
     "month, finance has no month-end numbers", 2),
    ("Photo uploads over the mobile app fail about half the time with a "
     "timeout, Android is unaffected", 2),
    ("Failed to publish the invoice: connection refused by the message broker", 2),
    ("Checkout submission aborted, the card processor rejected the batch", 2),
    ("The export job crashes whenever a customer name contains a non-Latin "
     "character, three enterprise tenants are blocked", 2),
    ("GitLab CI cannot reach the artifact store since the network change, all "
     "pipelines fail at the upload step", 2),
    ("The calendar integration silently drops recurring-event updates, "
     "double-bookings are being reported", 2),
    # CRITICAL: outages, data loss, unrecoverable failures.
    ("The entire billing pipeline is down and we are losing inbound payment "
     "events, finance estimates thousands of unprocessed transactions", 3),
    ("A bad migration wiped the audit-log table in production and we have no "
     "replica to restore from", 3),
    ("The eu-west cluster is completely unreachable, every service in the "
     "region is returning connection refused", 3),
    ("Customer data from the last six hours was written to the wrong tenant "
     "and cannot be un-mixed after the fact", 3),
    ("The primary database is corrupted beyond online repair and failover "
     "failed, engineering is declaring a full outage", 3),
    ("Node web-03 stopped responding, all in-flight carts on it are gone", 3),
    ("Replication is broken and the primary disk is dead, taking the cluster "
     "offline now", 3),
    ("Ransomware signatures were found on the build runners, all CI credentials "
     "are being rotated and the artifact cache is quarantined", 3),
    ("The control plane of the orchestrator is gone, no workload in any region "
     "can be scheduled or stopped", 3),
    ("Sev-declared: the single sign-on provider outage means no employee can "
     "access any internal system and customer SSO logins fail too", 3),
]


def _level_token_pattern():
    global _LEVEL_TOKEN
    if _LEVEL_TOKEN is None:
        alts = "|".join(re.escape(t) for t in ["INFO", "WARNING", "ERROR", "FATAL", "CRITICAL"])
        _LEVEL_TOKEN = re.compile(r"(?i)(?<![A-Za-z])(" + alts + r")(?![A-Za-z])")
    return _LEVEL_TOKEN


def load_loghub_grade_slices() -> dict[str, list[tuple[str, int]]]:
    """(text, gold_tier_index) pairs per slice from LogHub dev."""
    path = ROOT / "data" / "real" / "log_event" / "dev.jsonl"
    slices: dict[str, list[tuple[str, int]]] = {"implicit": [], "explicit": []}
    if not path.is_file():
        return slices
    token = _level_token_pattern()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        level = (row.get("gold") or {}).get("level")
        if not level:
            continue
        tier = SEVERITY_LEVEL_ALIAS.get(level, level)
        if tier not in SEVERITY_TIERS:
            continue
        text = row["raw"].strip()
        if not text:
            continue
        key = "explicit" if token.search(text) else "implicit"
        slices[key].append((text, SEVERITY_TIERS.index(tier)))
    return slices


# 3. Near-miss verify cases on BFCL-style (request, tool) pairs: the guard's
# home domain. Labels follow BFCL semantics: 1 = the tool serves the request,
# 0 = it does not. These are hand-written boundary conditions, not training
# rows (BFCL corpus is disjoint; near-miss phrasings are new).
# ``tool_dict`` (when present) feeds the deterministic fillability gate
# (tool_match.validate_fillability): the cross-encoder judges semantics ("is
# this the right kind of tool") but does not count parameters; the composite
# decision (verify AND fillable) is the shippable acceptance rule and is
# reported alongside the verify-only numbers.
NEAR_MISS_VERIFY_CASES = [
    # Parameter-count mismatch: sum needs two numbers, request supplies one.
    ("Calculate the sum of 42",
     "calculate_sum: Adds two numbers a and b and returns the total.", 0,
     {"name": "calculate_sum", "description": "Adds two numbers a and b and returns the total.",
      "parameters": {"properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                     "required": ["a", "b"]}}),
    ("Calculate the sum of 42 and 58",
     "calculate_sum: Adds two numbers a and b and returns the total.", 1,
     {"name": "calculate_sum", "description": "Adds two numbers a and b and returns the total.",
      "parameters": {"properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                     "required": ["a", "b"]}}),
    # Type conflict: tool expects a date, request supplies free text.
    ("Book me a table for dinner sometime soon",
     "find_restaurant: Books a table at a restaurant for a specific ISO date (YYYY-MM-DD).", 0,
     None),
    ("Book me a table for dinner on 2026-10-15",
     "find_restaurant: Books a table at a restaurant for a specific ISO date (YYYY-MM-DD).", 1,
     None),
    # Partial keyword overlap: shared vocabulary, wrong capability.
    ("I want to convert my US dollars to euros for my vacation",
     "get_currency_symbol: Returns the currency symbol string for a country code.", 0,
     None),
    ("What is the currency symbol for Japan?",
     "get_currency_symbol: Returns the currency symbol string for a country code.", 1,
     None),
    ("Transfer money from my savings account to my checking account",
     "get_exchange_rate: Returns the exchange rate between two currency codes.", 0,
     None),
    ("What is the exchange rate from USD to EUR today?",
     "get_exchange_rate: Returns the exchange rate between two currency codes.", 1,
     None),
    # Unrelated domain.
    ("Play my favorite jazz playlist",
     "calculate_sum: Adds two numbers a and b and returns the total.", 0,
     None),
]


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_select_multi_candidate(matcher: tool_match.BaseToolMatcher, n_items: int = 100) -> dict:
    """Benchmark select on held-out multi-candidate requests with distractors."""
    pairs_file = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
    if not pairs_file.is_file():
        return {"accuracy": 0.0, "total": 0, "p50_ms": 0.0, "p95_ms": 0.0}

    random.seed(20261006)
    dev_rows = [
        json.loads(line)
        for line in pairs_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("split") == "dev"
    ]
    pos_rows = [r for r in dev_rows if r["label"] == 1]
    neg_tools = list({r["tool"] for r in dev_rows if r["label"] == 0})

    latencies = []
    correct = 0
    total = min(n_items, len(pos_rows))

    # Warmup
    matcher.select("Warmup request", {"t1": "t1: test tool", "t2": "t2: another tool"})

    for r in pos_rows[:total]:
        gold_tool = r["tool"]
        gold_key = gold_tool.split(":")[0].strip()
        distractors = random.sample(neg_tools, 3)
        cands = {gold_key: gold_tool}
        for i, d in enumerate(distractors):
            k = d.split(":")[0].strip() or f"distractor_{i}"
            cands[k] = d

        t0 = time.perf_counter()
        choice, conf = matcher.select(r["request"], cands)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        if choice == gold_key:
            correct += 1

    return {
        "accuracy": (correct / total) * 100.0 if total else 0.0,
        "correct": correct,
        "total": total,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "mean_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def bench_prompt_routing(matcher: tool_match.BaseToolMatcher) -> dict:
    """Benchmark select for prompt-to-model routing across specialized domains."""
    latencies = []
    correct = 0
    total = len(PROMPT_ROUTING_DATASET)

    for prompt, expected in PROMPT_ROUTING_DATASET:
        t0 = time.perf_counter()
        choice, conf = matcher.select(prompt, PROMPT_ROUTING_MODELS)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        if choice == expected:
            correct += 1

    return {
        "accuracy": (correct / total) * 100.0 if total else 0.0,
        "correct": correct,
        "total": total,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def bench_routing_10_domains(matcher: tool_match.BaseToolMatcher) -> dict:
    """Adversarial routing: 10 domains x (clean, terse, messy, boundary)."""
    latencies = []
    per_condition: dict[str, list[int]] = {}
    wrong: list[tuple[str, str, str]] = []
    correct = 0
    total = len(ROUTING_DATASET_10)

    # Warmup happens in the other select benches.
    for prompt, expected, condition in ROUTING_DATASET_10:
        t0 = time.perf_counter()
        choice, conf, status = matcher.select(prompt, ROUTING_DOMAINS_10, with_status=True)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        hit = choice == expected
        correct += hit
        per_condition.setdefault(condition, []).append(hit)
        if not hit:
            wrong.append((prompt[:50], expected, choice))

    cond_stats = {}
    for cond, hits in sorted(per_condition.items()):
        cond_stats[cond] = sum(hits) / len(hits) * 100.0
    return {
        "accuracy": correct / total * 100.0 if total else 0.0,
        "correct": correct,
        "total": total,
        "per_condition": cond_stats,
        "wrong": wrong,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def bench_verify_near_miss(matcher: tool_match.BaseToolMatcher) -> dict:
    """verify on hand-written boundary pairs (parameter/type/keyword near-misses).

    Also reports the composite decision (verify AND fillable) for the cases
    that carry a tool dict: the semantic gate accepts parameter-starved
    requests, the deterministic fillability gate catches them.
    """
    latencies = []
    tp = fp = tn = fn = 0
    c_tp = c_fp = c_tn = c_fn = 0
    misses: list[str] = []
    composite_misses: list[str] = []
    for request, tool, label, tool_dict in NEAR_MISS_VERIFY_CASES:
        t0 = time.perf_counter()
        verified, score = matcher.verify(request, tool, threshold=0.50)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if label and verified:
            tp += 1
        elif label and not verified:
            fn += 1
            misses.append(f"FN score={score:.3f}: {request[:40]!r}")
        elif not label and not verified:
            tn += 1
        else:
            fp += 1
            misses.append(f"FP score={score:.3f}: {request[:40]!r}")
        if tool_dict is not None:
            fillable, reason, _missing = tool_match.validate_fillability(tool_dict, request)
            accepted = verified and fillable
            if label and accepted:
                c_tp += 1
            elif label:
                c_fn += 1
                composite_misses.append(f"FN reason={reason or 'verify'}: {request[:40]!r}")
            elif accepted:
                c_fp += 1
                composite_misses.append(f"FP score={score:.3f}: {request[:40]!r}")
            else:
                c_tn += 1
    total = len(NEAR_MISS_VERIFY_CASES)
    n_composite = sum(1 for c in NEAR_MISS_VERIFY_CASES if c[3] is not None)
    acc = (tp + tn) / total * 100.0 if total else 0.0
    prec = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
    rec = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
    c_acc = (c_tp + c_tn) / n_composite * 100.0 if n_composite else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "misses": misses,
        "composite": {
            "accuracy": c_acc,
            "tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
            "total": n_composite,
            "misses": composite_misses,
        },
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
    }


def bench_grade_novel_register(matcher: tool_match.BaseToolMatcher) -> dict:
    """grade on the pre-registered novel-register severity set (business/SRE
    phrasings, never trained on). This is the honest generalization metric.
    Also reports the tier-set decode (grade with_set=True) at the pinned
    quantile: coverage (gold inside the set) and mean set size — the set
    carries the "between tier i and i+1" uncertainty the argmax hides."""
    latencies = []
    exact = 0
    covered = 0
    sizes = []
    total = len(NOVEL_REGISTER_GRADE_CASES)
    for text, gold in NOVEL_REGISTER_GRADE_CASES:
        t0 = time.perf_counter()
        pred_anchor, _score, tier_set = matcher.grade(
            text, SEVERITY_ANCHORS_EXCLUSIVE, with_set=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        pred = {"info": 0, "warning": 1, "error": 2, "critical": 3}[pred_anchor.split(" ")[0]]
        exact += pred == gold
        covered += gold in tier_set
        sizes.append(len(tier_set))
    from mivia_small_json.stats import wilson_interval

    exact_ci = wilson_interval(exact, total)
    coverage_ci = wilson_interval(covered, total)
    return {
        "exact": exact / total * 100.0 if total else 0.0,
        "exact_ci": (round(exact_ci[0] * 100, 1), round(exact_ci[1] * 100, 1)),
        "correct": exact,
        "total": total,
        "set_coverage": covered / total * 100.0 if total else 0.0,
        "set_coverage_ci": (round(coverage_ci[0] * 100, 1),
                            round(coverage_ci[1] * 100, 1)),
        "covered": covered,
        "mean_set_size": float(np.mean(sizes)) if sizes else 0.0,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
    }


def bench_grade_loghub(matcher: tool_match.BaseToolMatcher) -> dict:
    """grade on real LogHub dev severity rows, both slices, both phrasings."""
    slices = load_loghub_grade_slices()
    phrasings = {
        "exclusive": SEVERITY_ANCHORS_EXCLUSIVE,
        "cumulative": SEVERITY_ANCHORS_CUMULATIVE,
    }
    out: dict[str, dict] = {}
    for slice_name, data in slices.items():
        if not data:
            continue
        for phrasing, anchors in phrasings.items():
            latencies = []
            exact = 0
            within1 = 0
            for text, gold in data:
                t0 = time.perf_counter()
                pred_anchor, _score = matcher.grade(text, anchors)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                pred = SEVERITY_TIERS.index(
                    pred_anchor.split(":")[0].strip().split(" ")[0].upper()
                )
                exact += pred == gold
                within1 += abs(pred - gold) <= 1
            out[f"{slice_name}/{phrasing}"] = {
                "exact": exact / len(data) * 100.0,
                "within1": within1 / len(data) * 100.0,
                "n": len(data),
                "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            }
    return out


def bench_verify_heldout(matcher: tool_match.BaseToolMatcher) -> dict:
    """Benchmark verify on all 1,706 held-out dev pairs in BFCL."""
    pairs_file = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
    if not pairs_file.is_file():
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "total": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }

    dev_rows = [
        json.loads(line)
        for line in pairs_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("split") == "dev"
    ]

    latencies = []
    tp, fp, tn, fn = 0, 0, 0, 0

    # Warmup
    matcher.verify("Warmup request", "tool: Warmup description")

    for r in dev_rows:
        req = r["request"]
        tool = r["tool"]
        expected = bool(r["label"])

        t0 = time.perf_counter()
        verified, score = matcher.verify(req, tool, threshold=0.50)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        if expected and verified:
            tp += 1
        elif expected and not verified:
            fn += 1
        elif not expected and not verified:
            tn += 1
        else:
            fp += 1

    total = len(dev_rows)
    acc = ((tp + tn) / total) * 100.0 if total else 0.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def bench_grade_ordinal(matcher: tool_match.BaseToolMatcher) -> dict:
    """Benchmark grade across ordinal capability and complexity anchors."""
    latencies = []
    correct = 0
    total = len(GRADE_DATASET)

    for prompt, expected in GRADE_DATASET:
        t0 = time.perf_counter()
        best_anchor, score = matcher.grade(prompt, GRADE_ANCHORS)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        pred_key = best_anchor.split(":")[0].strip()
        if pred_key == expected:
            correct += 1

    return {
        "accuracy": (correct / total) * 100.0 if total else 0.0,
        "correct": correct,
        "total": total,
        "p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier1-only",
        action="store_true",
        help="evaluate Tier 1 (MiniLM-L-12) only without Tier 2 ModernBERT",
    )
    args = parser.parse_args()

    tier1_path = ROOT / "data" / "incoming" / "models" / "tool-match-l12-int8.onnx"
    tier2_path = ROOT / "data" / "incoming" / "models" / "tool-match-modernbert-int8.onnx"

    if args.tier1_only:
        matcher = tool_match.load_matcher(tier1_path)
        name = "Tier 1 Only (MiniLM-L-12)"
    else:
        matcher = tool_match.load_matcher(
            tier1_path, tier2=tool_match.Tier2Spec(path=tier2_path)
        )
        name = "Cascaded Guard (Tier 1 -> Tier 2)"

    if matcher is None:
        print("Error: Failed to load matcher model.")
        return 1

    print("=" * 64)
    print(f"Zero-Shot Primitives Benchmark: {name}")
    print("=" * 64)

    rss_init = _rss_mb()

    # 1. Select
    res_select = bench_select_multi_candidate(matcher, n_items=100)
    res_prompt = bench_prompt_routing(matcher)
    res_routing10 = bench_routing_10_domains(matcher)
    print("\n[Primitive 1: select (Multi-Candidate & Prompt Routing)]")
    print(
        f"  Candidate Selection (1-of-4): {res_select['accuracy']:.1f}% "
        f"({res_select['correct']}/{res_select['total']}) | "
        f"p50: {res_select['p50_ms']:.2f} ms | p95: {res_select['p95_ms']:.2f} ms"
    )
    print(
        f"  Prompt-to-Model Routing:     {res_prompt['accuracy']:.1f}% "
        f"({res_prompt['correct']}/{res_prompt['total']}) | "
        f"p50: {res_prompt['p50_ms']:.2f} ms | p95: {res_prompt['p95_ms']:.2f} ms"
    )
    print(
        f"  10-Domain Adversarial:       {res_routing10['accuracy']:.1f}% "
        f"({res_routing10['correct']}/{res_routing10['total']}) | "
        f"p50: {res_routing10['p50_ms']:.2f} ms | p95: {res_routing10['p95_ms']:.2f} ms"
    )
    for cond, pct in res_routing10["per_condition"].items():
        print(f"    {cond:9s}: {pct:.1f}%")
    for prompt, expected, choice in res_routing10["wrong"]:
        print(f"    miss: want {expected}, got {choice}: {prompt!r}")

    # 2. Verify
    res_verify = bench_verify_heldout(matcher)
    correct_v = res_verify['tp'] + res_verify['tn']
    print("\n[Primitive 2: verify (1,706 Held-out BFCL Dev Pairs)]")
    print(f"  Accuracy:  {res_verify['accuracy']:.2f}% ({correct_v}/{res_verify['total']})")
    print(f"  Precision: {res_verify['precision']:.2f}%")
    print(f"  Recall:    {res_verify['recall']:.2f}%")
    print(f"  F1 Score:  {res_verify['f1']:.2f}%")
    print(f"  Latency:   p50: {res_verify['p50_ms']:.2f} ms | p95: {res_verify['p95_ms']:.2f} ms")

    res_near_miss = bench_verify_near_miss(matcher)
    nm_correct = res_near_miss["tp"] + res_near_miss["tn"]
    print(f"\n[Primitive 2b: verify (Near-Miss Boundaries, "
          f"{res_near_miss['total']} hand-written pairs)]")
    print(f"  Accuracy:  {res_near_miss['accuracy']:.1f}% ({nm_correct}/{res_near_miss['total']})")
    print(f"  Precision: {res_near_miss['precision']:.1f}% | "
          f"Recall: {res_near_miss['recall']:.1f}%")
    for m in res_near_miss["misses"]:
        print(f"    {m}")
    comp = res_near_miss["composite"]
    c_correct = comp["tp"] + comp["tn"]
    print(f"  Composite (verify AND fillable, {comp['total']} pairs with tool dicts): "
          f"{comp['accuracy']:.1f}% ({c_correct}/{comp['total']})")
    for m in comp["misses"]:
        print(f"    composite {m}")

    # 3. Grade — on the dedicated ordinal artifact when present (the guard
    # matcher's backbone has no ordinal training; the split is the measured
    # promotion decision, see reports/training-report.md).
    grade_matcher = tool_match.load_grade_matcher()
    grade_target = grade_matcher if grade_matcher is not None else matcher
    grade_name = "ordinal artifact" if grade_matcher is not None else "guard matcher"
    res_grade = bench_grade_ordinal(grade_target)
    print(f"\n[Primitive 3: grade (Ordinal Complexity Alignment, {grade_name})]")
    print(
        f"  Accuracy:  {res_grade['accuracy']:.1f}% "
        f"({res_grade['correct']}/{res_grade['total']}) | "
        f"p50: {res_grade['p50_ms']:.2f} ms | p95: {res_grade['p95_ms']:.2f} ms"
    )

    res_sev = bench_grade_loghub(grade_target)
    print(f"\n[Primitive 3b: grade (LogHub Severity, real held-out rows, {grade_name})]")
    for key, stats in res_sev.items():
        print(
            f"  {key:22s}: exact {stats['exact']:.1f}% | within-1 {stats['within1']:.1f}% "
            f"(n={stats['n']}) | p50: {stats['p50_ms']:.2f} ms"
        )

    res_novel = bench_grade_novel_register(grade_target)
    print(f"\n[Primitive 3c: grade (Novel Register, pre-registered "
          f"{res_novel['total']} cases, {grade_name})]")
    print(
        f"  Exact:     {res_novel['exact']:.1f}% ({res_novel['correct']}/{res_novel['total']}) "
        f"95% CI {res_novel['exact_ci']} | p50: {res_novel['p50_ms']:.2f} ms"
    )
    print(
        f"  Tier set:  coverage {res_novel['set_coverage']:.1f}% "
        f"({res_novel['covered']}/{res_novel['total']}) "
        f"95% CI {res_novel['set_coverage_ci']} | "
        f"mean size {res_novel['mean_set_size']:.2f} "
        f"(pinned quantile {tool_match.GRADE_SET_QUANTILE})"
    )

    rss_final = _rss_mb()
    print(f"\nPeak Memory RSS: {rss_final:.1f} MB (delta: +{rss_final - rss_init:.1f} MB)")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
