# mivia-small-json

Turn raw workflow records into strict JSON. Give it an issue, a log line, a commit message, a pull request, an AI agent tool call, a compiler diagnostic, a git diff, or a code review comment — or plain non-workflow text. Get a validated JSON object.

The system handles **11 router classes simultaneously** (10 workflow classes + plain `text` for non-workflow input) under hard 2 GB RAM, CPU-only constraints.

## State of the model (shipped bytes: `mm12-realworld-r1-s42`, mMiniLM-L12 multilingual, 11 classes)

Binding state: [docs/MODEL-STATE.md](docs/MODEL-STATE.md) · measured gaps: [docs/LIMITATIONS.md](docs/LIMITATIONS.md) · data policy: [docs/DATA.md](docs/DATA.md) · weights: [Hugging Face](https://huggingface.co/mivialabs).

**What it can do (measured on the shipped bytes):**

- **Routing**: 99.71% frozen held-out routed-and-valid (held across the whole mm12 lineage; independently reproduced: 99.7%, 0 spurious `text` predictions in-domain); LogHub 32k: routed 100.0% / valid 100% / component 99.99% / explicit level 99.98%; **SWE-bench 97/100 problem statements, 98/100 gold patches, validity 100%** (a ±6pp run-variance band exists on this surface — quote it); external hosted-router head-to-head routing **99.71 vs 97.14** on identical rows (independent verification).
- **Non-workflow input is a class, not a rejection**: plain-`text` emission with registered one-shot bars — rejection 90.3%, free-chat 94.5%, retention min **0.99** (tickets-non-en 1.0, tickets-de 0.99 — all PASS); the seven real-world routing slices are pre-registered floors (`tests/test_realworld_bars.py`), all PASS on the shipped bytes.
- **Extraction on real prose** (learned extractor): title exact **71.8% (en) / 67.5% (de)**, body token recall **~0.996** on held-out real tickets; Log-shaped extraction at ceiling: 99.99% component / 99.96% explicit level on LogHub; implicit syslog severity 77.5% on unseen systems; templates GA 0.887.
- **Tool-call guard** (honest abstention): 79.7% abstention at 95.1% retention on the canonical 5,100-row eval; `no_tool_needed` 96.6%; **serves 71.5% of real agent-session invocations (AUC 0.93)**; full-set catalogue AUC 0.9030.
- **Severity grading** (business register, production = register gate → specialist): 46.8% exact / **84.2% within-1** / QWK 0.675 on 600 real tickets (promoted specialist, +11.2pp, p=2.8e-16); canonical-register control 67.7% exact / 93.5% within-1.
- **Calibration & safety**: conformal budget for the guard, strict JSON validation with ≤2 repair retries, never-raise pipeline, offline CPU inference (p50 ~8–16 ms / p95 ~12–21 ms short text; 878 MB RSS).

**What it cannot do (measured gaps — do not assume otherwise):**

- **SWE-bench problem-statement routing carries a measured ±6pp run-variance band** (86 vs 97 across identical-config runs) — quote the band; n=100.
- **Guard residual categories**: infeasible 36.7% / sibling_swap 63.5% / ask_followup 74.8% — need architecture; the product outcome for unsure cases is `tool_clarification` (CLARIFY: ask the user for missing parameters).
- **Zero-shot select() over 10 short domain descriptions: 9/12** — a measured tension between select-contrast and real-session service in one rank head. Open-domain zero-shot baselines: Banking77 **48.0% top-1** over 77 intents (n=600) with the wired select-rank head at **81.0%** (docs/LIMITATIONS.md), JevBench v1 **54.1%**, S1MB 35.5% floor.
- **BFCL tool-call routing is 88.0%** (161/183) routed-to-expected on the public 183-row set — misses abstain honestly; the 90.7% elsewhere is the distinct corpus-v2 rejection rate (both named in MODEL-STATE).
- **SWE-bench title extraction is 7.2%** (body recall 0.986) — long problem statements lack a crisp title sentence.
- **Severity middle tiers on ambiguous real text**: tier 2 at 9.6% exact remains weakest; label quality (kappa 0.717) caps exact-match scores.
- **Long-input latency is load-dependent, not a model property**: a load-inflated "2× slower" claim was measured and WITHDRAWN (quiet box ~63–87 ms at 11k chars); the encoder caps at 512 tokens by design and input capping is deliberately not applied.

For in-depth developer examples and pattern breakdowns, see [docs/USAGE.md](docs/USAGE.md). For end-to-end enterprise architectures (LLM adapters, agent telemetry, CI/CD triage, log ingestion, air-gapped deployment), see [docs/USE_CASES.md](docs/USE_CASES.md).

## Install

Requires Python 3.12+.

1. Run `python3 -m venv .venv`.
2. Run `.venv/bin/pip install -r requirements.txt`.
3. Run `.venv/bin/pip install -e .`.
4. Assemble the split model artifacts (the >100 MB ONNX files ship as
   byte-split parts to stay under GitHub's file-size limit):
   `.venv/bin/python scripts/assemble_models.py`.

The Docker path below does the assembly for you. Training/fine-tuning
code needs the optional GPU stack: `pip install -e ".[training]"` —
runtime inference never needs it.

## Serve it (HTTP + MCP, Docker)

Deploy the model over the internet in one container — REST quartet, zero-shot primitives, and an official-protocol MCP server on one port, bearer-token auth included:

```bash
docker build -t mivia-small-json .
docker run -d -p 8000:8000 -e MIVIA_API_TOKEN=change-me mivia-small-json

curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/guard \
  -H 'Authorization: Bearer change-me' -H 'Content-Type: application/json' \
  -d '{"request": "calculate the sum of 15 and 30",
       "tools": [{"name": "calculate_sum",
                  "description": "Calculate the sum of two numbers.",
                  "parameters": {"type": "object",
                                 "properties": {"num1": {"type": "number"}, "num2": {"type": "number"}},
                                 "required": ["num1", "num2"]}}]}'
# -> {"action": "CALL", "tool": "calculate_sum", ...}  (CLARIFY / ABSTAIN for the honest cases)
```

MCP clients connect to `http://HOST:8000/mcp` (streamable HTTP; `claude mcp add --transport http mivia http://HOST:8000/mcp --header "Authorization: Bearer change-me"`). Without Docker: `pip install -e ".[serve]"` then `python -m mivia_small_json.api --host 0.0.0.0 --port 8000`. The stdlib-only sidecar (`python -m mivia_small_json.serve`, `python -m mivia_small_json.mcp`) stays available for air-gapped, zero-extra-dependency embedding. Full deployment guide: [docs/USE_CASES.md §10](docs/USE_CASES.md).

## Use from the command line

1. Run `echo "ERROR [auth]: token expired" | .venv/bin/python -m mivia_small_json.infer`.
2. Read one JSON line per input record on stdout:
   ```json
   {"input": "ERROR [auth]: token expired", "obj": {"type": "log_event", "level": "ERROR", "component": "auth", "message": "token expired", "ts": "..."}, "meta": {"routed_by": "model", "valid": true, "retries": 0, "latency_ms": 1.05}}
   ```

## Use from Python

Register the schemas into your process, load the SHA-256 verified router, and classify:

```python
from mivia_small_json import bootstrap, diagnostic, diff_comment, pipeline, tool_call

# 1. Register schemas into the process registry
bootstrap.register_example_schemas()             # pull_request
tool_call.register_tool_call_schema()            # tool_call
diagnostic.register_diagnostic_schema()          # diagnostic
diff_comment.register_diff_and_review_schemas()  # diff, review_comment

# 2. Load the verified production router (int8 ONNX encoder; the skops
#    fallback below is what you get if the assembled ONNX artifact or the
#    onnxruntime package is unavailable)
router = pipeline.load_active_router()
# router = pipeline.load_router("models/router.skops")  # fallback

# 3. Classify an atomic record
obj, meta = pipeline.classify("ERROR [auth]: token expired", router)
```

`obj` holds the validated record. `meta` indicates how it was produced: `routed_by` (`"model"`, `"rules"`, `"structure"` — the deterministic log-shape decision layer, or `"template"`), `valid` (strict validator pass), `retries` (repair attempts, <= 2), and `latency_ms`.

### Multi-line log streams and conversational dumps (`classify_dump`)

Real-world developer logs often contain conversational preambles (e.g. *"hey what is this error?"*) and multi-line stack traces. Use `classify_dump(text)`:

```python
dump = """hey what is this error?
Jun 14 15:16:01 combo sshd[19939]: authentication failure; rhost=1.2.3.4
Jun 14 15:16:02 combo sshd[19937]: check pass; user unknown
Jun 15 02:04:59 combo sshd[20882]: session closed for user root
"""

records = pipeline.classify_dump(dump, router)
for obj, meta in records:
    print(obj["type"], obj["level"], obj["component"], obj["message"])
```

`classify_dump` automatically:
- Filters conversational preambles.
- Groups continuation lines and multi-line stack traces (`at ...`, `Caused by:`, leading tabs/spaces) with their parent log line.
- Preserves atomic multi-line entities (`diff`, `diagnostic`, `review_comment`, `issue`, `pull_request`, `commit`) as single records.

### Intent selection (`/v1/select`)

Zero-shot pick over a candidate set. Default `head: "tool"` is the shipped
guard (79.7% abstention @ 95.1% retention on its canonical eval). For
intent-shaped label sets, pass `"head": "intent"`: a language-gated dual
head routes English text to the English intent head (82.8-85% top-1 on
Banking77 fresh reserve) and everything else to the multilingual head
(62% on the 8-language dev split where the English head scores 10%) —
kappa-0.959 agent-authored training corpus (not redistributed; built by
`scripts/build_intent_pairs.py` per `data/real/intent-pairs/PROVENANCE.md`).

### 3-State Tool Routing & Clarification (AI Agent Workflows)

Before executing tools or invoking slow frontier models, agent runtimes route the user prompt and tool catalogue through `mivia-small-json` (p95 ~12–21 ms on CPU; the guard chain is a 23 MB gate + 23 MB rank int8 ONNX pair). The engine classifies into three deterministic states:

1. **State 1 (`tool_call`)**: High match with complete arguments; healed and strictly validated JSON ready for immediate dispatch.
2. **State 2 (`tool_clarification`)**: High topical alignment with a tool, but mandatory parameters are missing (e.g., asking *"calculate the sum"* without numbers). Emits structured `missing_parameters` and a user-ready `clarification_prompt` so the agent asks the user instead of hallucinating or crashing.
3. **State 3 (`no_tool_call`)**: Abstention when the request is out-of-domain or irrelevant to listed tools; agent continues regular conversation.

```python
import json
from mivia_small_json import pipeline

# 1. Format input with available tools and user request
prompt = f"Tools available: {json.dumps(tools)}\n\nUser request: {user_query}"

# 2. Fast CPU classification (< 15 ms)
result, meta = pipeline.classify(prompt)

# 3. Dispatch based on state
if result["type"] == "tool_call":
    execute_tool(result["name"], json.loads(result["arguments"]))
elif result["type"] == "tool_clarification":
    ask_user(result["clarification_prompt"])
elif result["type"] == "no_tool_call":
    reply_with_chat(user_query)
```

See the complete guide in [docs/TOOL_ROUTING_AND_CLARIFICATION.md](docs/TOOL_ROUTING_AND_CLARIFICATION.md).


### Zero-Shot Decision Primitives & Free Ticket Triage (`select`, `grade`, `verify`)

Replace expensive cloud LLM calls for classification, routing, and triage with three CPU-native zero-shot discriminative primitives running in **10–25 ms** at **$0.00 cost**:

```python
from mivia_small_json.tool_match import load_matcher

matcher = load_matcher()

# 1. Multi-Candidate Selection: route support ticket to team zero-shot
team, conf = matcher.select(
    "Checkout hangs on 3D Secure with HTTP 504",
    candidates=["Frontend UI", "Payments & Billing Core", "Auth", "DevOps"]
)  # -> ("Payments & Billing Core", 0.96) in ~12 ms

# 2. Ordinal Severity Grading: assign SLA priority
severity, sev_score = matcher.grade(
    "Checkout hangs on 3D Secure with HTTP 504",
    anchors=["P4 - Low typo", "P3 - Workaround exists", "P2 - Degraded", "P1 - Outage"]
)  # -> ("P1 - Outage", 0.94) in ~10 ms

# 3. Binary Entailment Verification: calibrated truth check
needs_escalation, verify_score = matcher.verify(
    "Checkout hangs on 3D Secure with HTTP 504",
    hypothesis="A production payment gateway is failing for active customers."
)  # -> (True, 0.99) in ~8 ms
```

See the full enterprise guide with full ticket triage architectures in [docs/USE_CASES.md](docs/USE_CASES.md).


### Custom schema registration

Your application can register custom schemas using `registry.register_schema()`:

```python
from mivia_small_json import registry, pipeline

registry.register_schema(
    "release",
    spec={"type": {"const": "release"}, "version": {"type": str}},
    routing={"strong": [r"(?i)\brelease\b"], "weak": []},
    extraction={"version": [r"v(\d+\.\d+\.\d+)"]},
    defaults={"version": "v0.0.0"},
)
obj, meta = pipeline.classify("release v1.2.3 shipped today", None)
```

The committed model routes only its trained classes. Pass `router=None` for rule-based routing, or retrain the model after you register formats.

## Output formats

The system supports 12 schemas out of the box:

| Type | Fields | Notes |
|---|---|---|
| `issue` | `type`, `id`, `title`, `labels`, `priority`, `state` | `priority`: low/medium/high/critical; `state`: open/closed |
| `log_event` | `type`, `level`, `component`, `message`, `ts` | `level`: TRACE/DEBUG/INFO/WARN/ERROR/FATAL (OTel); `ts`: ISO 8601 |
| `commit` | `type`, `sha`, `message`, `files_changed`, `author`, `commit_type` | `commit_type` is optional (Conventional Commits: feat, fix, chore, docs, style, refactor, perf, test, build, ci) |
| `pull_request` | `type`, `number`, `title`, `author`, `state`, `additions`, `deletions` | `state`: open/merged/closed/draft |
| `tool_call` | `type`, `name`, `arguments`, `result` | AI agent function calling & tool execution traces; `arguments` is strict JSON |
| `tool_clarification` | `type`, `tool`, `missing_parameters`, `reason`, `clarification_prompt` | Proactive user clarification when mandatory tool arguments are omitted |
| `no_tool_call` | `type`, `reason` | Safe abstention when user input is out-of-domain or irrelevant to listed tools |
| `diagnostic` | `type`, `tool`, `file`, `line`, `column`, `severity`, `code`, `message` | Compiler/linter diagnostics (rustc, tsc, ruff, gcc, clang, pytest); `severity`: error/warning/note |
| `diff` | `type`, `old_path`, `new_path`, `change_type`, `additions`, `deletions`, `patch` | Unified git diffs; `change_type`: add/modify/delete/rename |
| `review_comment` | `type`, `file`, `line`, `comment`, `severity`, `suggested_code` | Code review comments & suggestions; `severity`: nit/suggestion/warning/blocking; `suggested_code` is optional |
| `agent_step` | `type`, `step_index`, `agent_role`, `tool`, `arguments`, `stdout`, `stderr`, `return_code`, `duration_ms`, `error_type`, `error_message`, `status` | Multi-turn agent trajectory steps; `agent_role`: assistant/tool/user/environment; `status`: success/error/timeout/interrupted |
| `text` | `type`, `reason` | Non-workflow input: the general-text outcome (product decision — no rejection wording anywhere) |

The output always matches one of these schemas. A validator verifies every field. A repair step fixes invalid values (<= 2 retries).


## Behavior

- `classify()` and `classify_dump()` never raise. Broken input returns a best-effort object and `valid: false` in `meta`.
- The model file carries a SHA-256 sidecar. The loader checks it first. A missing or corrupt file safely falls back to rule-based routing.
- Average inference latency: p50 ~8–16 ms / p95 ~12–21 ms on CPU for short text (the shipped multilingual encoder is a 118 MB int8 ONNX + 17 MB tokenizer; the rules fallback is sub-millisecond).
- Model load time: ~0.6 s once per process (sha-verified ONNX + tokenizer).

## Limits

- The rules router knows the built-in schemas and any registered formats. Other text still produces schema-shaped output. Check `meta.valid` before trusting values.
- Field values are extracted via patterns and learned classifiers. A value falls back to schema defaults when the input does not match.

## Data

Run `python -m mivia_small_json.synth` to regenerate the built-in synthetic corpus (the training/eval synthetic sets that ship in `data/synthetic/` and `data/heldout/`).

**Third-party training and evaluation data is not redistributed.** The model was trained and measured on public corpora (LogHub, BFCL, Glaive function-calling, Banking77, MASSIVE, GitHub-sourced issues/commits/PRs, and others) whose licenses do not all permit redistribution in an Apache-2.0 repository. Instead, every source is documented — upstream URL, license, admission rules, row counts, content sha256 — in the provenance records under `data/`, and each corpus has a deterministic builder script under `scripts/` that re-fetches and rebuilds it from upstream. Reproducibility is a recipe, not a blob.

Benchmark numbers quoted in this README were measured on those upstream datasets; fetch them per the provenance instructions to reproduce. The LogHub-based results additionally credit the [LogHub](https://github.com/logpai/loghub) (logpai) maintainers for collecting and publishing those log datasets.

The repository can also improve itself from your own data. Drop real records into `data/pool/inbox/`, then run `python -m mivia_small_json.flywheel --train`. The loop collects, weak-labels them with the rules layer, retrains over the pool, and evaluates on both frozen sets. A gate matrix decides: a pull request opens when every check passes; otherwise you get a report only.

## Benchmark

### External LogPai LogHub Benchmark (16 Systems, 32,000 Lines)

The full pipeline (model routing + learned component/level extraction + validation) evaluated against LogHub ground truth (`_structured.csv`, 2,000 lines per system, 16 systems, 32,000 total lines; dataset fetched on demand — see docs/DATA.md).

| Metric | Result | Target |
|---|---|---|
| **Routing Accuracy** | **99.98%** (31,996 / 32,000) | >= 99.0% |
| **Valid JSON Rate** | **100.0%** (32,000 / 32,000) | >= 99.0% |
| **Component Accuracy** | **99.99%** (29,996 / 30,000 across 15 systems with ground truth) | >= 95.0% |
| **Explicit Level Accuracy** | **99.96%** (17,993 / 18,000 across 9 systems with explicit levels) | >= 95.0% |

Systems evaluated: Android, Apache, BGL, HDFS, HPC, Hadoop, HealthApp, Linux, Mac, OpenSSH, OpenStack, Proxifier, Spark, Thunderbird, Windows, and Zookeeper.

### Tool-call Guard & Severity Grading

| Surface | Result |
|---|---|
| Guard, canonical 9-category eval (5,100 rows) | **79.7% abstention @ 95.1% retention**; catalogue AUC 0.9030 |
| Guard vs an external hosted router | full-set catalogue AUC **0.9030**; matched-sample separation 0.8374 vs 0.8451 (the external router marginally ahead) but it structurally cannot hold a 95%-retention operating point (0.01-quantized outputs) — mivia holds 37% abstention @ 96.3% on that sample |
| Guard honesty scoping | the 95%-retention operating point is BFCL-catalogue-shaped; on real agent-session catalogues the promoted session-transfer rank serves **71.5% of invocations** (AUC 0.93; honest pre-promotion baseline 21.7% — an earlier "99.7%" was a constant-score artifact) |
| Severity, real business tickets (n=600) | **46.8% exact / 84.2% within-1 / QWK 0.675** (promoted specialist; +11.2pp, p=2.8e-16) |
| Severity, canonical-register control (n=217) | 67.7–70.0% exact |
| Router calibration (n=2,800) | **ECE 0.0017** |
| Field extraction on real tickets/PRs | learned prose extractor: title 71.8/67.5% en/de exact, body recall ~0.996; weakest surface SWE-bench titles 7.2% |
| Reality check: routing on real data | strong — **97% SWE-bench (±6pp run-variance band)**, 98.4/92.1% tickets (en/de), **80.3% real PRs**, 98.3% review comments |

## Models on Hugging Face

The shipped weights are published on the Hugging Face Hub:

- [mivialabs/mivia-small-json-router](https://huggingface.co/mivialabs/mivia-small-json-router) -
  the 11-class router encoder: int8 ONNX, tokenizer, class map, skops fallback.
- [mivialabs/mivia-small-json-tool-match](https://huggingface.co/mivialabs/mivia-small-json-tool-match) -
  the tool-match guard: rank + gate int8 ONNX with tokenizers.

Both are grouped in the
[mivia-small-json collection](https://huggingface.co/collections/mivialabs/mivia-small-json-6acb7817285b7e7c4d9b45a8).
Each model card carries its own usage snippet and evaluation table.

## For contributors and agents

Read [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before
you change this repository. Run the focused test files for the surfaces you
touch, and `pytest -q` once before you commit.

## License

[Apache-2.0](LICENSE). Model weights are fine-tuned from third-party base
checkpoints — attribution in [NOTICE.md](NOTICE.md). "MiviaLabs", "mivia", and
"mivia.app" are trademarks of MiviaLabs and are not licensed under the
Apache License; the hosted API lives at
https://api.mivia.app/v1/models/mivia-small-json and is a separate product
from this repository.
