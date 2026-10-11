# Developer Guide: Using mivia-small-json

`mivia-small-json` transforms unstructured software development signals into strictly-validated JSON records under strict 2 GB RAM, CPU-only constraints. The router's final class map is 11 classes — the 10 workflow schemas below plus the general `text` class (chat, news, invoices: a first-class outcome, never a rejection):

- `issue`
- `commit`
- `log_event`
- `pull_request`
- `tool_call`
- `tool_clarification`
- `diagnostic`
- `diff`
- `review_comment`
- `agent_step`
- `text`

plus the defensive `no_tool_call` abstention schema kept for the guard's no-tool outcome (the router itself no longer emits it).

**No-truncation contract:** output content is never truncated. Every extracted field — log and commit messages, diagnostic messages (all lines of a multi-line compiler traceback included), agent-step error messages, sha ids — carries the full matched text back to you; nothing is sliced, first-lined, or padded into a placeholder (a too-short hex sha is reported absent, never zero-padded into fabricated digits). Records stay bounded because your input is bounded: each request accepts up to 256 KiB of text, and a record can never exceed what you sent. If you have a larger dump, split it on line boundaries before sending — every line you send comes back.

> **Enterprise Architectures & Pipelines**: For full end-to-end production architectures (LLM adapters, AI agent telemetry, compiler/CI pipelines, multi-line syslog ingestion, and air-gapped deployments), see [docs/USE_CASES.md](USE_CASES.md).

---

## 1. Quick Setup & Loading

### Schema Registration

The core repository contains built-in schemas (`issue`, `commit`, `log_event`) and extensible registered schemas. To activate all 8 schemas in your process, call their registration functions:

```python
from mivia_small_json import bootstrap, diagnostic, diff_comment, pipeline, tool_call

# Register schemas into the runtime registry (idempotent)
bootstrap.register_example_schemas()                 # registers "pull_request"
tool_call.register_tool_call_schema()                # registers "tool_call"
tool_call.register_no_tool_call_schema()             # registers "no_tool_call"
tool_call.register_tool_clarification_schema()        # registers "tool_clarification"
diagnostic.register_diagnostic_schema()              # registers "diagnostic"
diff_comment.register_diff_and_review_schemas()      # registers "diff" and "review_comment"
```

### Loading the Model Router

The router is loaded using `pipeline.load_router()`. It verifies the SHA-256 sidecar checksum before loading the `skops` artifact, falling back safely to deterministic rules if the file is missing or corrupt:

```python
from mivia_small_json.model import MODEL_PATH
from mivia_small_json.pipeline import load_router

# Load the sha256-verified skops router
router = load_router(MODEL_PATH)  # or pass "models/router.skops"
```

---

## 2. Single-Record Classification (`classify`)

`pipeline.classify()` accepts any string and returns an `(obj, meta)` tuple. It **never raises an exception**, even on hostile or malformed input.

```python
from mivia_small_json import pipeline

raw_text = "ERROR [auth]: token expired for user 1042"
obj, meta = pipeline.classify(raw_text, router)

print("Parsed record:", obj)
print("Metadata:", meta)
```

**Output:**
```python
obj = {
    "type": "log_event",
    "level": "ERROR",
    "component": "auth",
    "message": "token expired for user 1042",
    "ts": "2026-10-05T13:46:24.000000Z"
}

meta = {
    "routed_by": "model",   # "model", "rules", or "template"
    "valid": True,          # True only if strictly valid before template fallback
    "retries": 0,           # Number of repair passes executed (<= 2)
    "latency_ms": 1.05      # Execution latency in milliseconds
}
```

---

## 3. Multi-Line Dumps & Log Streams (`classify_dump`)

Real-world developer logs are rarely clean single-line strings. Developers paste chat preambles, multi-line logs, and complex stack traces.

`pipeline.classify_dump()`:
1. Strips conversational preambles (e.g., *"hey what is this error?"*).
2. Groups continuation lines and multi-line stack traces (lines starting with spaces, tabs, `at `, `Caused by:`, or `...`) with their parent log line.
3. Separates individual log events into discrete records.
4. Preserves multi-line atomic entities such as Git diffs and compiler diagnostics intact.

```python
from mivia_small_json import pipeline

raw_dump = """hey what is this error?
Jun 14 15:16:01 combo sshd[19939]: authentication failure; rhost=1.2.3.4
Jun 14 15:16:02 combo sshd[19937]: check pass; user unknown
Jun 15 02:04:59 combo sshd[20882]: session closed for user root
"""

records = pipeline.classify_dump(raw_dump, router)

for obj, meta in records:
    print(f"[{obj['level']}] ({obj['component']}) {obj['message']}")
```

**Output:**
```
[ERROR] (sshd) authentication failure; rhost=1.2.3.4
[WARN] (sshd) check pass; user unknown
[INFO] (sshd) session closed for user root
```

### Multi-Line Stack Trace Example

```python
stack_dump = """ERROR [database]: Connection refused
\tat com.example.db.Pool.connect(Pool.java:42)
\tat com.example.db.Client.query(Client.java:108)
Caused by: java.net.ConnectException: Connection refused: no further information
"""

records = pipeline.classify_dump(stack_dump, router)
obj, meta = records[0]

print("Type:", obj["type"])
print("Component:", obj["component"])
print("Message contains trace:", "Connection refused: no further information" in obj["message"])
```

---

## 4. Compiler & Linter Diagnostics (`diagnostic`)

`mivia-small-json` structures errors and warnings from `rustc`, `tsc`, `ruff`, `gcc`, `clang`, `pytest`, and `mypy` following SARIF and Language Server Protocol (LSP) conventions.

```python
rustc_error = """error[E0308]: mismatched types
  --> src/main.rs:12:18
   |
12 |     let x: u32 = "hello";
   |                  ^^^^^^^ expected `u32`, found `&str`
"""

obj, meta = pipeline.classify(rustc_error, router)
print(obj)
```

**Output:**
```python
{
    "type": "diagnostic",
    "tool": "rustc",
    "file": "src/main.rs",
    "line": 12,
    "column": 18,
    "severity": "error",
    "code": "E0308",
    "message": "mismatched types"
}
```

### Sanitizing Terminal Output (`sanitize_log`)

Before parsing terminal output or CI runner logs that contain ANSI escape codes, carriage-return progress bars, or CI metadata wrappers, use `sanitize_log`:

```python
from mivia_small_json.diagnostic import sanitize_log

raw_ci_output = "\x1b[31;1m2026-10-05T03:14:07.123Z ##[error]src/api.ts(45,7): error TS2322: Type 'string' is not assignable to type 'number'.\x1b[0m"
clean = sanitize_log(raw_ci_output)
obj, meta = pipeline.classify(clean, router)

print("Clean text:", clean)
print("File:", obj["file"], "Line:", obj["line"], "Code:", obj["code"])
```

---

## 5. AI Agent Tool Calls, Abstention & Clarification (`tool_call`, `tool_clarification`, `no_tool_call`)

Autonomous AI agents (CodeAct, SWE-agent, OpenHands, OpenAI/Anthropic tool use) emit and process function calls and execution traces. Rather than executing tools blindly or hallucinating missing parameters, agent runtimes route the request and tool catalog through `mivia-small-json` (< 15 ms on CPU) before dispatch.

For a dedicated architectural reference, see [docs/TOOL_ROUTING_AND_CLARIFICATION.md](TOOL_ROUTING_AND_CLARIFICATION.md).

### The 3-State Tool Routing Architecture

```
User Prompt + Tool Catalog
          │
          ▼
mivia-small-json (< 15 ms CPU)
          │
  ┌───────┼────────────────────────┐
  ▼       ▼                        ▼
State 1: tool_call       State 2: tool_clarification       State 3: no_tool_call
(Valid JSON arguments;   (High match, missing args;        (Chat / no matching tool;
 execute immediately)     prompt user for parameters)       continue normal conversation)
```

1. **State 1 (`tool_call`)**: High alignment and complete parameters. Arguments are repaired and validated into strict JSON ready for execution. For tools whose required parameters are all strings, the arguments are filled **deterministically from the request text itself** — key-delimited (`service=nginx`, `"host": "web-01"`) and alignment (`host is web-01`) ladders return the exact verbatim spans, with no model in the loop and no invented values; a slot the request text cannot prove never gets a guess (it stays on the clarification path).
2. **State 2 (`tool_clarification`)**: High topical alignment with a registered tool, but mandatory arguments are missing (e.g. asking *"calculate the sum"* without numbers or *"fetch URL"* without an address). The model emits structured `missing_parameters`, `reason`, and a user-ready `clarification_prompt` so the agent asks the user instead of hallucinating.
3. **State 3 (`no_tool_call`)**: Abstention when the request matches none of the listed tools, continuing normal conversational chat.

#### String-Parameter Filling Example

```python
import json
from mivia_small_json import pipeline, tool_call

tool_call.register_tool_call_schema()
tool_call.register_tool_clarification_schema()
tool_call.register_no_tool_call_schema()
router = pipeline.load_active_router()

tools = [{
    "type": "function",
    "function": {
        "name": "restart_service",
        "description": "Restart a service on a host.",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "host": {"type": "string"}
            },
            "required": ["service", "host"]
        }
    }
}]

# The request carries both values verbatim -> CALL with the exact spans
req = f"Tools available: {json.dumps(tools)}\n\nUser request: restart the stuck service. service=nginx host=web-01"
result, meta = pipeline.classify(req, router=router)
# result = {"type": "tool_call", "name": "restart_service",
#           "arguments": "{\"service\":\"nginx\",\"host\":\"web-01\"}", "result": ""}

# The request carries no value for the slot -> the honest minimum: a CALL
# whose arguments carry nothing ("{}") — the value is never invented; the
# /v1/guard surface additionally downgrades such calls to a structured
# tool_clarification when the fillability veto confirms the request
# supplies nothing.
req2 = f"Tools available: {json.dumps(tools)}\n\nUser request: restart the stuck service"
result2, meta2 = pipeline.classify(req2, router=router)
# result2 = {"type": "tool_call", "name": "restart_service", "arguments": "{}", "result": ""}
```

### Input Protocol

Requests format the tool catalog and user intent in the Berkeley Function Calling Leaderboard (BFCL) format:

```text
Tools available: [{"name": "fetch_url", "description": "Fetch web content", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}]

User request: Download data from https://api.example.com/v1/metrics
```

### Tri-State Payloads

#### State 1 Payload: `tool_call`
```json
{
  "type": "tool_call",
  "name": "fetch_url",
  "arguments": "{\"url\":\"https://api.example.com/v1/metrics\"}",
  "result": ""
}
```

#### State 2 Payload: `tool_clarification`
```json
{
  "type": "tool_clarification",
  "tool": "calculate_sum",
  "missing_parameters": ["numbers"],
  "reason": "Missing required argument 'numbers' for tool 'calculate_sum'",
  "clarification_prompt": "Please specify the list of numbers you would like to sum."
}
```

#### State 3 Payload: `no_tool_call`
```json
{
  "type": "no_tool_call",
  "reason": "no matching tool"
}
```

### Agent Loop Switching Example

```python
import json
from mivia_small_json import pipeline, tool_call

# Register tool schemas into the runtime registry (idempotent)
tool_call.register_tool_call_schema()
tool_call.register_no_tool_call_schema()
tool_call.register_tool_clarification_schema()

def run_agent_step(tools_catalog: list[dict], user_prompt: str) -> str:
    # 1. Format the input string
    input_text = f"Tools available: {json.dumps(tools_catalog)}\n\nUser request: {user_prompt}"

    # 2. Fast CPU classification (< 15 ms, never raises)
    result, meta = pipeline.classify(input_text)

    state = result.get("type")

    # 3. Switch on the 3 states
    if state == "tool_call":
        tool_name = result["name"]
        args = json.loads(result["arguments"])
        return f"Executing {tool_name}({args})..."

    elif state == "tool_clarification":
        # Prompt user directly with the clarification question
        return result["clarification_prompt"]

    elif state == "no_tool_call":
        # Fall back to standard conversational chat
        return "I can answer your question directly without calling any tools."

    return "Processing conversation..."
```

### Trace Parsing & Argument Healing (`repair_arguments`)

For post-execution traces (`<functioncall> ... </functioncall>` or `FUNCTION RESPONSE:`), `tool_call` extracts the execution record:

```python
agent_trace = """<functioncall> {"name": "fetch_user", "arguments": '{"user_id": 42}'}
FUNCTION RESPONSE: {"status": 200, "user": "Alice"}
"""

obj, meta = pipeline.classify(agent_trace)
# obj = {"type": "tool_call", "name": "fetch_user", "arguments": "{\"user_id\":42}", "result": "{\"status\": 200, \"user\": \"Alice\"}"}
```

LLMs also frequently emit malformed pseudo-JSON arguments (single quotes, unquoted keys, Python literals `True`/`False`/`None`, or trailing commas). The `repair_arguments` helper normalizes them into strictly valid JSON:

```python
from mivia_small_json.tool_call import repair_arguments

malformed = "{'user_id': 101, 'active': True, 'tags': ['admin',],}"
repaired = repair_arguments(malformed)
print(repaired)
# Output: '{"user_id":101,"active":true,"tags":["admin"]}'
```

---


## 6. Git Diffs & Code Review Comments (`diff`, `review_comment`)

### Git Diffs

Unified patches are routed to the `diff` schema:

```python
diff_patch = """diff --git a/src/main.py b/src/main.py
index 1234567..89abcde 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,3 +10,4 @@
-old_implementation()
+new_implementation()
+verify_result()
"""

obj, meta = pipeline.classify(diff_patch, router)
print("Type:", obj["type"])
print("Paths:", obj["old_path"], "->", obj["new_path"])
print("Change type:", obj["change_type"])
```

For direct syntactic analysis of unified diff hunks (calculating exact addition and deletion counts from lines), use `extract_diff`:

```python
from mivia_small_json.diff_comment import extract_diff

diff_info = extract_diff(diff_patch)
print(f"Additions: +{diff_info['additions']}, Deletions: -{diff_info['deletions']}")
```

### Code Review Comments

Inline pull request review comments and code suggestion blocks are routed to `review_comment`:

```python
review_text = """LGTM with a small nit: rename variable `res` to `result`.
```suggestion
result = compute_total()
```
"""

obj, meta = pipeline.classify(review_text, router)
print(obj)
```

**Output:**
```python
{
    "type": "review_comment",
    "file": "",
    "line": 1,
    "comment": "LGTM with a small nit: rename variable `res` to `result`.",
    "severity": "nit",
    "suggested_code": "result = compute_total()"
}
```

---

## 7. Custom Schema Registration

You can extend `mivia-small-json` with domain-specific schemas using `registry.register_schema()`.

```python
from mivia_small_json import pipeline, registry

# Register a custom schema
registry.register_schema(
    name="release_event",
    spec={
        "type": {"const": "release_event"},
        "version": {"type": str},
        "channel": {"type": str, "enum": ("stable", "beta", "nightly")},
        "published": {"type": bool},
    },
    routing={
        "strong": [r"(?i)\brelease\b", r"\bv\d+\.\d+\.\d+\b"],
        "weak": [r"(?i)\bchannel\b", r"(?i)\bversion\b"],
    },
    extraction={
        "version": [r"\bv(\d+\.\d+\.\d+)\b"],
        "channel": [r"(?i)\bchannel[:=\s]+(stable|beta|nightly)\b"],
    },
    defaults={
        "version": "0.0.0",
        "channel": "stable",
        "published": True,
    },
)

# Route and parse with the rules engine (router=None)
raw = "release v2.4.1 channel: beta published"
obj, meta = pipeline.classify(raw, router=None)

print(obj)
```

**Output:**
```python
{
    "type": "release_event",
    "version": "2.4.1",
    "channel": "beta",
    "published": True
}
```

### 7.1 Enum Aliases — Declared Vocabulary Coercion

Real-world text does not use your enum's vocabulary. An enum field may
declare an optional **`aliases`** map: surface words — any language, any
vocabulary — that coerce to a canonical enum member.

```python
registry.register_schema(
    name="order",
    spec={
        "type": {"const": "order"},
        "id": {"type": str},
        "address": {"type": str},
        "email": {"type": str},
        "urgency": {
            "type": str,
            "enum": ["high", "normal", "low"],
            "aliases": {
                "high": ["urgent", "urgently", "asap", "as soon as possible",
                         "eilt", "dringend", "sürgős", "urgente",
                         "فوری", "عاجل", "срочно", "紧急", "긴급"],
                "low": ["whenever", "no rush", "low priority"],
            },
        },
    },
    routing={"strong": [r"(?i)\b(ship|order)\b"], "weak": []},
    extraction={
        "id": [r"(?i)\border\s+(\d+)\b"],
        # Non-greedy stop: the address terminates at "as soon", so the
        # multi-word alias never leaks into the address span.
        "address": [r"(?i)\bto\s+(.+?)\s+as\s+soon\b"],
        "email": [r"(?i)\b([\w.+-]+@[\w-]+(?:\.[\w-]+)+)\b"],
        # Capture the FULL surface (single- or multi-word) up to the
        # next punctuation — the alias lookup matches the whole span.
        "urgency": [r"(?i)\bit(?:'s|\s+is)\s+(.+?)(?=[.,;!?]|$)"],
    },
)

obj, meta = pipeline.classify(
    "Hi, please ship order 8841 to 12 Oak Rd as soon as you can, it is "
    "urgent. You can reach me at a@x.io.", router=None)
# obj == {"type": "order", "id": "8841", "address": "12 Oak Rd",
#         "email": "a@x.io", "urgency": "high"}
```

**Coercion contract.** The captured span is normalized (trim +
casefold + wrap-strip: leading quotes and trailing punctuation are
wrapper noise), then matched EXACTLY: an enum member first, else a
declared alias; the canonical member is emitted. There is NO fuzzy
matching and NO engine-invented synonyms — only the schema author's
declared mapping ever rewrites a value. Validation is unchanged:
records carry canonical members; a raw alias surface in a record is
still an enum error.

**Refusal contract (coercion honesty).** When extraction FOUND a value
but coercion refuses it — no member and no declared alias matches — the
field is treated as UNFILLED: it flows through the missing-field path
and stays **absent** in the emitted record. It is never repaired to a
nearest-looking member and never silently filled with the default. The
declared default applies ONLY when extraction found nothing. If you
want refusal-to-absent, declare the enum field `"optional": True` (the
same convention as the shipped `commit_type`: *absent rather than
empty*); a required enum field with an unmatched value falls through to
the author's declared default via the missing-field path, so prefer
optionality or a wider declared vocabulary.

Registration rules: `aliases` requires `enum` on the same rule (str
type); keys must be declared enum members; values are non-empty lists
of strings; a surface that normalizes to two different members is
rejected at registration. The map is defensively copied, like the rest
of the spec.

#### Starter multilingual vocabulary (copy and ADOPT into your schema)

The block below is a **starter vocabulary for schema AUTHORS** — a
convenience to copy into your own `aliases` declaration. It is **not**
engine behavior: nothing below coerces unless YOUR schema declares it.

```python
SEVERITY_ALIASES = {
    "high": [
        # en
        "urgent", "urgently", "asap", "as soon as possible",
        # de
        "eilt", "dringend",
        # fr
        "au plus vite",
        # es / pt
        "cuanto antes", "o quanto antes",
        # ro / hu
        "cât mai curând", "sürgős",
        # ar / fa
        "عاجل", "على وجه السرعة", "فوری", "در اسرع وقت",
        # ru
        "срочно", "как можно скорее",
        # zh / ja / ko
        "紧急", "加急", "尽快", "緊急", "至急", "긴급", "긴함",
    ],
    "low": [
        # en
        "whenever", "no rush", "low priority",
        # de / fr
        "keine eile", "pas urgent", "sans urgence",
        # es / pt
        "sin prisa", "sem pressa",
        # ro / hu
        "fără grabă", "nincs sietség",
        # ar / fa
        "بدون استعجال", "بدون عجله",
        # ru
        "не срочно", "без спешки",
        # zh / ja / ko
        "不急", "随时", "いつでも", "급하지 않음",
    ],
}
```

### 7.2 Severity vocabularies — the taxonomy crosswalk

Your tools speak PagerDuty, syslog, CVSS; the trained grade head does
not. It measures ONE canonical ordinal scale — the four anchors in
`severity_head.CANONICAL_SEVERITY_ANCHORS` (info → warning → error →
critical; tier 4 = most severe; the 0-based `tier_index` returned by
`triage_severity` is `tier - 1`). On the n=600 real-ticket v3 set the
production grade surface measures **46.8% exact / 84.2% within-1**
(`docs/MODEL-STATE.md`).

The crosswalk below is **not** a new measured model surface: it is a
deterministic, stdlib-only token map
(`mivia_small_json.severity_taxonomies`) from ten production
vocabularies onto those four tiers — so a tier-3 grade can render as
SEV-2 / P2 / major in the customer's own vocabulary, and (with §7.1
aliases) the customer's tokens coerce onto your canonical enum at
extraction time. Per-taxonomy polarity is the whole point: SEV-1 is
the *worst* PagerDuty can say, while syslog `0` (emergency) is the
worst on its scale and OTel's worst is SeverityNumber 24. Each
mapping below names its source; judgment calls are recorded in place.
Regenerate the generated block with
`python scripts/gen_severity_alias_packs.py` — never hand-edit it.

<!-- BEGIN severity-taxonomies: generated by scripts/gen_severity_alias_packs.py from src/mivia_small_json/severity_taxonomies.py — regenerate, never hand-edit -->
The trained grade head measures the CANONICAL scale only — 46.8% exact / 84.2% within-1 on the n=600 real-ticket v3 set (docs/MODEL-STATE.md). The crosswalk is NOT a measured model surface: it is deterministic token mapping, stdlib-only, no model bytes.

Tier names below are the canonical anchor names (``severity_head.CANONICAL_SEVERITY_ANCHORS``); **tier 4 is most severe**. The model's 0-based ``tier_index`` (``triage_severity``, ``severity_head.cumulative_tier``) is ``tier - 1``.

| Taxonomy | Tier 1 `info` | Tier 2 `warning` | Tier 3 `error` | Tier 4 `critical` |
|---|---|---|---|---|
| `pagerduty` — PagerDuty SEV (incident/ops) | SEV-5 | SEV-4 | SEV-2 · SEV-3 | SEV-0 · SEV-1 |
| `atlassian_p` — Atlassian / Jira P1-P4 (incident/ops) | P4 | P3 | P2 | P1 |
| `bugzilla` — Bugzilla severities (bug tracking) | trivial · triv · enhancement · enh | normal · nor · minor · min | major · maj | blocker · block · critical · crit |
| `jira_priority` — Jira priorities (bug tracking) | lowest | medium · low | high | highest |
| `github_labels` — GitHub label conventions (bug tracking) | low · P3 | medium · P2 | high · P1 | critical · P0 |
| `syslog` — syslog RFC 5424 (logs) | 5-7 · notice · informational · info · debug | 4 · warn · warning | 1-3 · alert · crit · critical · err · error | 0 · emerg · emergency · panic |
| `otel` — OTel SeverityNumber 1-24 (logs) | 1-12 · TRACE · DEBUG · INFO | 13-16 · WARN · WARNING | 17-20 · ERROR | 21-24 · FATAL |
| `log4j` — Log4j levels (logs) | TRACE · DEBUG · INFO | WARN · WARNING | ERROR | FATAL |
| `cvss` — CVSS qualitative severity (security) | low · none | medium | high | critical |
| `itil` — ITIL priority = impact x urgency (ITSM) | P4 · low · P5 · very low | P3 · moderate · medium | P2 · high | P1 · critical |

**Sources and judgment calls** (named, per the repo's honesty rules):

- `pagerduty` — PagerDuty Incident Response Docs, 'Severity Levels' (response.pagerduty.com, repo PagerDuty/incident-response-docs): lower number = more severe; 'Anything above a SEV-3 is automatically considered a "major incident"'; 'All SEV-2's are major incidents'.
  - SEV-0 -> tier 4: informal extension: orgs that use SEV-0 mean 'worse than SEV-1'; there is no slot above canonical tier 4
  - SEV-1 -> tier 4: PagerDuty SEV-1 = critical issue: system in a critical state, actively impacting a large number of customers, customer-data-exposing vulnerability -> tier 4 (severe failure / data loss requiring immediate action)
  - SEV-2 -> tier 3: PagerDuty SEV-2 = critical system issue actively impacting many customers (a major incident, short of SEV-1) -> tier 3 (failures that disrupted service). Judgment call: PD's own SEV-2 wording can describe an app unavailable for most/all users; tier 4 is reserved for SEV-1-grade complete outage / data loss — re-pin SEV-2 -> tier 4 if your SEV-2 IS a full outage
  - SEV-3 -> tier 3: PagerDuty SEV-3 = partial loss of functionality, not affecting the majority of customers -> tier 3 (a failure that disrupted the task, for part of the user base)
  - SEV-4 -> tier 2: PagerDuty SEV-4 = minor issues requiring action but not affecting customer ability to use the product (performance delays, single-node failure, cron failure) -> tier 2 (recoverable abnormal conditions, not failures)
  - SEV-5 -> tier 1: PagerDuty SEV-5 = cosmetic issues or bugs -> tier 1 (routine)
- `atlassian_p` — Atlassian incident-severity guidance (atlassian.com, 'Understanding incident severity levels'): SEV 1 'a critical incident with very high impact', SEV 2 'a major incident with significant impact', SEV 3 'a minor incident with low impact'; the extended model adds minor-usability / cosmetic tiers. Rendered here in the common P1-P4 incident-priority notation (lower = more severe).
  - P1 -> tier 4: Atlassian P1 = critical incident, very high impact, all hands -> tier 4
  - P2 -> tier 3: Atlassian P2 = major incident with significant impact (short of P1) -> tier 3
  - P3 -> tier 2: Atlassian P3 = minor incident with low impact -> tier 2
  - P4 -> tier 1: Atlassian extended model P4 = minor usability / cosmetic impact -> tier 1
- `bugzilla` — Bugzilla severity field values (bugzilla.mozilla.org / bugzilla.org documentation): blocker, critical, major, normal, minor, trivial, enhancement; the legacy short forms block/crit/maj/nor/min/triv/enh are the same values.
  - blocker / block -> tier 4: blocks development and/or testing or the release -> tier 4 (immediate-action blocker)
  - critical / crit -> tier 4: Bugzilla critical = crashes, loss of data, severe memory leak -> tier 4 (the canonical critical anchor's own words: severe crashes, data loss)
  - major / maj -> tier 3: major loss of function -> tier 3 (failures that disrupted a task or transaction)
  - normal / nor -> tier 2: the ordinary default defect severity -> tier 2 (an abnormal condition needing attention, not an operational failure)
  - minor / min -> tier 2: minor loss of function, easy workaround exists -> tier 2
  - trivial / triv -> tier 1: cosmetic problems (misspellings, alignment) -> tier 1
  - enhancement / enh -> tier 1: request for enhancement — not a defect at all -> tier 1 (routine)
- `jira_priority` — Jira / Jira Service Management default priority scheme: Highest (address immediately), High (address soon), Medium (in the normal course), Low (when time permits), Lowest (at the assignee's discretion).
  - highest -> tier 4: address immediately -> tier 4
  - high -> tier 3: address soon to mitigate real damage -> tier 3
  - medium -> tier 2: in the normal course of work -> tier 2 (needs attention, not a failure)
  - low -> tier 2: when time permits — still a tracked abnormality, just not urgent -> tier 2
  - lowest -> tier 1: at the assignee's discretion -> tier 1 (routine)
- `github_labels` — GitHub community label conventions: priority:/severity: word labels (critical/high/medium/low) and the P0-P3 convention (P0 = drop-everything critical ... P3 = low).
  - critical -> tier 4: label severity critical -> tier 4
  - P0 -> tier 4: P0 convention: drop everything, site-level emergency -> tier 4
  - high -> tier 3: label severity high -> tier 3
  - P1 -> tier 3: P1 convention: urgent, important -> tier 3
  - medium -> tier 2: label severity medium -> tier 2
  - P2 -> tier 2: P2 convention: normal priority -> tier 2
  - low -> tier 1: label severity low -> tier 1
  - P3 -> tier 1: P3 convention: low priority -> tier 1
- `syslog` — RFC 5424 §6.2.1 Severity (0 = Emergency ... 7 = Debug; LOWER numeric value = MORE severe), mapped level-by-level per the OTel logs data-model Appendix B table: Emergency->FATAL(21), Alert->ERROR3(19), Critical->ERROR2(18), Error->ERROR(17), Warning->WARN(13), Notice->INFO2(10), Informational->INFO(9), Debug->DEBUG(5).
  - 0 / emerg / emergency / panic -> tier 4: syslog 0 Emergency -> OTel FATAL band (SeverityNumber 21) -> tier 4. Emergency-vs-FATAL debate, for the record: RFC 5424 Emergency means 'system is unusable' and is often emitted by a monitor that survives, while application-fatal means the emitting process is dying — some argue Emergency should therefore NOT be fatal-tier. OTel's table nonetheless puts Emergency in the FATAL band and we follow the documented mapping. 'panic' is the legacy syslog(3) name for the same level
  - 1 / alert -> tier 3: syslog 1 Alert -> OTel ERROR3 (19) -> tier 3: OTel deliberately documents Alert inside the ERROR band, not FATAL — tier 4 is Emergency-only on this taxonomy
  - 2 / crit / critical -> tier 3: syslog 2 Critical ('critical conditions') -> OTel ERROR2 (18) -> tier 3. NAME COLLISION, deliberate: canonical tier 4 is *named* 'critical', but the crosswalk follows the documented OTel placement (ERROR band), not the English word — see the alias-pack caveat in docs/USAGE.md §7.2
  - 3 / err / error -> tier 3: syslog 3 Error -> OTel ERROR (17) -> tier 3 (operation failures)
  - 4 / warn / warning -> tier 2: syslog 4 Warning -> OTel WARN (13) -> tier 2
  - 5 / notice -> tier 1: syslog 5 Notice ('normal but significant') -> OTel INFO2 (10) -> tier 1: OTel documents Notice inside the INFO band (a stricter reading would put it in tier 2; we follow the documented mapping)
  - 6 / informational / info -> tier 1: syslog 6 Informational -> OTel INFO (9) -> tier 1
  - 7 / debug -> tier 1: syslog 7 Debug -> OTel DEBUG (5) -> tier 1
- `otel` — OpenTelemetry logs data-model, SeverityNumber: bands TRACE(1-4), DEBUG(5-8), INFO(9-12), WARN(13-16), ERROR(17-20), FATAL(21-24); HIGHER number = MORE severe (opposite of syslog's numeric direction).
  - TRACE / 1 / 2 / 3 / 4 -> tier 1: SeverityNumber 1-4, TRACE band -> tier 1
  - DEBUG / 5 / 6 / 7 / 8 -> tier 1: SeverityNumber 5-8, DEBUG band -> tier 1
  - INFO / 9 / 10 / 11 / 12 -> tier 1: SeverityNumber 9-12, INFO band -> tier 1
  - WARN / WARNING / 13 / 14 / 15 / 16 -> tier 2: SeverityNumber 13-16, WARN band -> tier 2
  - ERROR / 17 / 18 / 19 / 20 -> tier 3: SeverityNumber 17-20, ERROR band -> tier 3
  - FATAL / 21 / 22 / 23 / 24 -> tier 4: SeverityNumber 21-24, FATAL band -> tier 4
- `log4j` — Log4j built-in levels; the OTel Appendix B table maps TRACE->1, DEBUG->5, INFO->9, WARN->13, ERROR->17, FATAL->21. OFF and ALL are logger CONTROLS, not severities: deliberately unmapped (resolve() returns None for them).
  - TRACE -> tier 1: Log4j TRACE -> tier 1
  - DEBUG -> tier 1: Log4j DEBUG -> tier 1
  - INFO -> tier 1: Log4j INFO -> tier 1
  - WARN / WARNING -> tier 2: Log4j WARN -> tier 2
  - ERROR -> tier 3: Log4j ERROR -> tier 3
  - FATAL -> tier 4: Log4j FATAL -> tier 4
- `cvss` — CVSS v3.1/v4.0 Qualitative Severity Rating Scale (None, Low, Medium, High, Critical) plus the standard vulnerability-management CVSS->priority mapping (Critical->P1, High->P2, Medium->P3, Low->P4).
  - critical -> tier 4: CVSS Critical (9.0-10.0) -> P1 -> tier 4
  - high -> tier 3: CVSS High (7.0-8.9) -> P2 -> tier 3
  - medium -> tier 2: CVSS Medium (4.0-6.9) -> P3 -> tier 2
  - low -> tier 1: CVSS Low (0.1-3.9) -> P4 -> tier 1 (the vuln-management convention puts Low in the lowest ops tier)
  - none -> tier 1: CVSS None (0.0): no security impact -> tier 1 (routine)
- `itil` — ITIL incident-management priority matrix: priority = impact x urgency over the 3x3 grid -> P1 critical (high impact x high urgency), P2 high (high x med or med x high), P3 moderate (high x low, med x med, low x high), P4 low (med x low, low x med), P5 very low (low x low).
  - P1 / critical -> tier 4: P1 = high impact x high urgency -> tier 4
  - P2 / high -> tier 3: P2 = high x medium or medium x high -> tier 3
  - P3 / moderate / medium -> tier 2: P3 = moderate (high x low, medium x medium, low x high) -> tier 2
  - P4 / low -> tier 1: P4 = low (medium x low, low x medium) -> tier 1
  - P5 / very low -> tier 1: P5 = low x low -> tier 1

#### Alias packs (generated from the tables above)

Paste into your enum field's `aliases` (§7.1). The canonical members are the four tier names; every surface coerces EXACTLY (no fuzzy matching). Surfaces that equal a canonical member name (syslog `critical`, log4j `error`, `warning`, `info`) are omitted — they already coerce as members; serve such taxonomies with distinct member names and pass them as `members=` to `alias_pack()`. To combine packs, use `merge_packs()` — a naive `{**a, **b}` merge REPLACES each member's list and silently drops the first pack. A surface that means two tiers across packs (jira `low` = tier 2 vs CVSS/GitHub/ITIL `low` = tier 1) is refused by `merge_packs()` (and by the registry at registration) BY DESIGN — pick the pack that matches your tool, or reconcile deliberately.

```python
# PagerDuty SEV (incident/ops)
PAGERDUTY_SEVERITY_ALIASES = {
    "info": ["SEV-5"],
    "warning": ["SEV-4"],
    "error": ["SEV-2", "SEV-3"],
    "critical": ["SEV-0", "SEV-1"],
}

# Atlassian / Jira P1-P4 (incident/ops)
ATLASSIAN_P_SEVERITY_ALIASES = {
    "info": ["P4"],
    "warning": ["P3"],
    "error": ["P2"],
    "critical": ["P1"],
}

# Bugzilla severities (bug tracking)
BUGZILLA_SEVERITY_ALIASES = {
    "info": ["trivial", "triv", "enhancement", "enh"],
    "warning": ["normal", "nor", "minor", "min"],
    "error": ["major", "maj"],
    "critical": ["blocker", "block", "crit"],
}

# Jira priorities (bug tracking)
JIRA_PRIORITY_SEVERITY_ALIASES = {
    "info": ["lowest"],
    "warning": ["medium", "low"],
    "error": ["high"],
    "critical": ["highest"],
}

# GitHub label conventions (bug tracking)
GITHUB_LABELS_SEVERITY_ALIASES = {
    "info": ["low", "P3"],
    "warning": ["medium", "P2"],
    "error": ["high", "P1"],
    "critical": ["P0"],
}

# syslog RFC 5424 (logs)
SYSLOG_SEVERITY_ALIASES = {
    "info": ["5", "notice", "6", "informational", "7", "debug"],
    "warning": ["4", "warn"],
    "error": ["1", "alert", "2", "crit", "3", "err"],
    "critical": ["0", "emerg", "emergency", "panic"],
}

# OTel SeverityNumber 1-24 (logs)
OTEL_SEVERITY_ALIASES = {
    "info": ["TRACE", "1", "2", "3", "4", "DEBUG", "5", "6", "7", "8", "9", "10", "11", "12"],
    "warning": ["WARN", "13", "14", "15", "16"],
    "error": ["17", "18", "19", "20"],
    "critical": ["FATAL", "21", "22", "23", "24"],
}

# Log4j levels (logs)
LOG4J_SEVERITY_ALIASES = {
    "info": ["TRACE", "DEBUG"],
    "warning": ["WARN"],
    "critical": ["FATAL"],
}

# CVSS qualitative severity (security)
CVSS_SEVERITY_ALIASES = {
    "info": ["low", "none"],
    "warning": ["medium"],
    "error": ["high"],
}

# ITIL priority = impact x urgency (ITSM)
ITIL_SEVERITY_ALIASES = {
    "info": ["P4", "low", "P5", "very low"],
    "warning": ["P3", "moderate", "medium"],
    "error": ["P2", "high"],
    "critical": ["P1"],
}

```

Round-trip (works end-to-end with §7.1 coercion):

```python
from mivia_small_json import pipeline, registry
from mivia_small_json.severity_taxonomies import merge_packs

registry.register_schema(
    name="incident",
    spec={
        "type": {"const": "incident"},
        "title": {"type": str},
        "severity": {
            "type": str,
            "enum": ["info", "warning", "error", "critical"],
            "aliases": merge_packs("pagerduty", "bugzilla"),
            "optional": True,  # honest refusal-to-absent choice (§7.1)
        },
    },
    routing={"strong": [r"(?i)\bincident\b"], "weak": []},
    extraction={
        "title": [r"(?i)\bincident:\s*(.+?)(?:;\s*severity\b|$)"],
        "severity": [r"(?i)\bseverity[\s:]+([A-Za-z0-9_-]+)"],
    },
)

obj, _ = pipeline.classify(
    "incident: checkout down for all users; severity: SEV-1", router=None)
# obj["severity"] == "critical"  (SEV-1 -> tier 4 member)
obj2, _ = pipeline.classify(
    "incident: typo on the pricing page; severity: trivial", router=None)
# obj2["severity"] == "info"      (bugzilla trivial -> tier 1 member)
```
<!-- END severity-taxonomies -->

### 7.3 Few-Shot Onboarding — teach a schema with ~8-16 labeled examples

When your schema's fields cannot be captured by regexes (free-text
slots, semantic categories, paraphrase), register it WITH labeled
examples and opt into onboarding: a small per-schema classifier is
trained in CPU-seconds and wired UNDER the deterministic extraction —
patterns stay authoritative when they fire, the learned head fills what
they leave open.

```python
from mivia_small_json import pipeline, registry

examples = [
    {"_text": "Where is my order of 2 x mechanical keyboard?",
     "type": "shop_order", "action": "track", "product": "mechanical keyboard",
     "quantity": 2},
    {"_text": "Please cancel my order for the laptop stand, quantity 1.",
     "type": "shop_order", "action": "cancel", "product": "laptop stand",
     "quantity": 1},
    # ... ~8-16 per enum class; every str/int value must appear VERBATIM
    # in its own text (that is the honesty contract, enforced on
    # training AND inference)
]
report = registry.register_schema(
    name="shop_order",
    spec={
        "type": {"const": "shop_order"},
        "action": {"type": str, "enum": ["place", "cancel", "modify", "track"]},
        "product": {"type": str},
        "quantity": {"type": int},
    },
    routing={"strong": [r"\border\b"], "weak": []},
    extraction={},               # all fields learned
    examples=examples,
    fewshot=True,                # <- the opt-in
)
# report == {"schema": "shop_order", "trained": True, "n_examples": 64, ...}

obj, _ = pipeline.classify("Has my order of 3 x usb-c dock shipped?",
                           router=None)
# obj["action"] == "track", obj["product"] == "usb-c dock" — learned, and
# the product is a VERBATIM span of the input or the field stays absent.
```

**How it works.** Enum fields (str + `enum`) get a message-level
LogisticRegression over the shipped sentence encoder's embedding; the
head can only emit a DECLARED enum member (your closed vocabulary — the
same trust level as §7.1 aliases, which keep applying to the
deterministic path). str/int fields get a token-level tagger (the same
FeatureHasher + LogisticRegression pattern as the shipped log
extractors); its decode is a contiguous span of the ORIGINAL text,
re-verified verbatim before it is emitted — a predicted value that does
not appear in the text is refused and the field stays absent. List
fields are never learned.

**Opt-in on purpose.** The shipped schemas register WITH examples on
every boot; `fewshot=True` is the explicit switch. Without it the
registration path is byte-identical to before (the feature is strictly
additive; the golden fixture and spec fingerprint are untouched).

**Lifecycle.** Artifacts persist to `data/fewshot/<schema>.skops`
(+ sha256 sidecar; `MIVIA_FEWSHOT_DIR` overrides the directory). The
wired head lives in memory; after a restart call
`fewshot.attach("shop_order")` to resume it from disk (sha-verified),
or just re-register — onboarding retrains in CPU-seconds anyway.
`registry.unregister("shop_order")` deletes the artifacts and the head.

**The measured guidance (F2 curve instrument).**
Exact-match on held-out rows at 2/4/8/16 examples per class:

| domain | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| b77-cards (6 intents, Banking77) | 0.628 | 0.792 | 0.826 | 0.883 |
| b77-transfers (6 intents, Banking77) | 0.656 | 0.747 | 0.804 | 0.840 |
| clinc-bank (6 intents, CLINC150) | 0.934 | 0.917 | 0.929 | 0.936 |
| clinc-travel (6 intents, CLINC150) | 0.990 | 0.982 | 0.992 | 0.995 |
| clinc-services (6 intents, CLINC150) | 0.986 | 0.996 | 0.996 | 0.995 |
| synthetic order (6 fields, str+int slots) | — | — | — | 0.963 |

Reading: semantically separated categories (travel, services, banking
tasks) onboard to >= 93% with as few as 2-4 examples per class, while
fine-grained near-synonym taxonomies (Banking77's transfer/card
sub-intents) keep climbing at 16 and want more examples or a coarser
enum. Mean over the five real domains at 16 examples/class:
**92.98%** (pre-registered bar >= 90%, spec
`few-shot-onboarding`; instrument
`.scratch/fewshot/measure_f1.py`, JSON + failed lines committed as
`docs/MODEL-STATE.md`).

**Cost.** Onboarding at 16 examples/class measured 0.1 s per intent
schema and 34.2 s for an 11-field worst case (fresh process, encoder
cold; bar 120 s). Memory stays inside the 2 GB envelope (RSS ~1.0 GB
with the encoder session loaded — the same profile as the shipped
guard).

---

## 8. Command-Line Inference

The CLI reads raw lines from `stdin` and writes JSON Lines (JSONL) to `stdout`:

```bash
# Classify an event
echo "ERROR [auth]: token expired" | python -m mivia_small_json.infer

# Process a file of mixed signals
cat input_records.txt | python -m mivia_small_json.infer > output.jsonl

# Specify an explicit model artifact path
cat input_records.txt | python -m mivia_small_json.infer --model models/router.skops
```

Each line on `stdout` is a complete JSON object formatted as:
```json
{
  "input": "ERROR [auth]: token expired",
  "obj": {
    "type": "log_event",
    "level": "ERROR",
    "component": "auth",
    "message": "token expired",
    "ts": "..."
  },
  "meta": {
    "routed_by": "model",
    "valid": true,
    "retries": 0,
    "latency_ms": 1.05
  }
}
```
Blank lines are skipped. Because of `classify`'s never-raise contract, hostile lines still yield valid records and the command always exits cleanly with code 0.

---

## 9. Zero-Shot Decision Primitives & Free Ticket Triage

Instead of paying for expensive cloud LLM calls (GPT-4o, Claude) to categorize text, grade severity, or perform sanity checks, `mivia-small-json` provides three CPU-native zero-shot discriminative primitives that run locally in **10–25 milliseconds** at **$0.00 cost**:

### 9.1 Multi-Candidate Selection (`select`)
Categorize unstructured text into arbitrary dynamic choices without retraining:

```python
from mivia_small_json.tool_match import load_matcher

matcher = load_matcher()

ticket = "Checkout page hangs during payment with HTTP 504 Gateway Timeout in EU region."

team, confidence = matcher.select(
    ticket,
    candidates=[
        "Frontend UI",
        "Payments & Billing Core",
        "Authentication",
        "DevOps & Infrastructure",
    ]
)
# Returns: ("Payments & Billing Core", 0.96) in ~12 ms
```

#### Description enrichment: pass examples, not just labels (measured +15.8pp)

Candidates may also be `{"name", "description", "examples"}` dicts. Appending
2-3 example utterances per label to the candidate's description is the single
measured lever for zero-shot custom-domain selection — **Banking77 zero-shot:
48.0% bare, 63.8% with per-label examples (600-row sample), and 92.5%
precision at 20% coverage when gated by confidence**:

```python
candidates = [
    {
        "name": "card_payment_not_showing",
        "description": "Card payment not showing on account",
        "examples": [
            "I paid my card last week and it is still not showing",
            "where is my payment",
        ],
    },
    {
        "name": "exchange_rate",
        "description": "Exchange rate and currency conversion questions",
        "examples": ["what is the gbp to eur rate today"],
    },
]

choice, confidence, status = matcher.select(ticket, candidates,
                                            with_status=True)
# The same dict form is served by POST /v1/select (head="tool";
# <= 64 candidates per request, the endpoint's standing cap).
```

Examples render with the measured separator convention —
`"<name>: <description>. Examples: \"<e1>\" \"<e2>\""` — so the string form
and the dict form are byte-identical for the same content. 2-3 examples per
label is the measured sweet spot; the endpoint caps examples at 8 per
candidate and 1024 chars each (the encoder window cannot read more).

#### Production mode: select-when-confident, abstain honestly otherwise

`with_status=True` adds an honest `status`: `"ok"` / `"ambiguous"` /
`"abstain"`. The abstain rule is conformal on the calibrated confidence, and
the calibration meta registers the tau per **candidate-count band** (refit
A single global tau — fitted on the 2-4-candidate shape —
abstained on EVERY many-candidate row, measured 600/600 on a 77-way
catalogue; the served tau now follows the candidate count, so the rule is
honest from a 2-way pick to a 151-way catalogue). The shipped production
mode for custom domains TODAY is confidence-gated selection: act when
`status == "ok"`, ask a human (or re-ask the user) otherwise. Measured
accuracy/coverage on the 600-row zero-shot benches: Banking77 bare
**77.8% @ 30% coverage, 88.3% @ 10%**; with per-label examples **92.5% @
20%**; CLINC150 (151-way) is stronger (85.6% @ 30%). `status == "abstain"`
is the honest no-decision — the guarantee is conditional on text resembling
the calibration population, so treat a called row as a high-precision
answer and an abstained row as a routing question, not a failure.

### 9.2 Ordinal Severity & SLA Grading (`grade`)
Grade incident descriptions against ordered anchor definitions:

```python
severity, score = matcher.grade(
    ticket,
    anchors=[
        "P4 - Low priority typo or cosmetic defect",
        "P3 - Minor bug with simple existing workaround",
        "P2 - Feature degraded but primary workflow operable",
        "P1 - Critical revenue-impacting outage or data corruption",
    ]
)
# Returns: ("P1 - Critical revenue-impacting outage...", 0.94) in ~10 ms
```

### 9.3 Calibrated Binary Entailment Verification (`verify`)
Perform a calibrated truth check with SelectiveNet rejection gating:

```python
is_escalation, score = matcher.verify(
    ticket,
    hypothesis="A production payment gateway is failing for active customers."
)
# Returns: (True, 0.99) in ~8 ms
```

