# AGENTS.md — model-mivia-small-json

Instructions for AI coding agents working in this repository. Humans: start
with [README.md](README.md).

## What this repo is

First model of the **mivia-small** family: `mivia-small-json` — small CPU
models that structure coding-workflow signals into strictly valid JSON.

- **11-Class Schema Router**: routes unstructured text across the 10
  workflow classes (`issue`, `commit`, `log_event`, `pull_request`,
  `tool_call`, `tool_clarification`, `diagnostic`, `diff`, `review_comment`,
  `agent_step`, plus `no_tool_call` — the legacy no-tool outcome, kept as a
  schema for the guard's `/v1/guard` abstention; only the router never
  emits it) plus plain `text` for non-workflow input — no rejection wording
  anywhere. The active router is a **fine-tuned int8 ONNX mMiniLM-L12
  multilingual encoder** (`models/router-encoder-int8.onnx`);
  `models/router.skops` is the hashed fallback when the ONNX artifact is
  absent.
- **Tool-match guard (`tool_match.py`)**: when the router routes a text to
  `tool_call`, a fine-tuned int8 ONNX cross-encoder scores the request
  against each listed tool. Below `THRESHOLD` the honest answer is
  `no_tool_call`. This is a relation a bag-of-ngrams model cannot express.
  The scorer must pass the tokenizer's segment ids; zeroing them costs ~55
  points of false positives.
- **Learned extraction (`extract_learned.py`)**: token- and message-level
  scikit-learn models for structured field extraction — explicit level,
  component, and implicit syslog severity.
- **Multi-line stream parsing (`pipeline.classify_dump`)**: strips
  conversational preambles, groups continuation lines and stack traces
  (`at ...`, `Caused by:`) with their parent log line, and preserves atomic
  multi-line entities (`diff`, `diagnostic`, `issue`).
- **Serving (`api.py`, `serve.py`, `mcp.py`)**: REST + MCP with bearer
  auth, a Host allow-list, opt-in per-key metering, and a bounded inference
  gate; a stdlib-only sidecar covers zero-dependency embedding.
- **Family roadmap**: `json → log → issue → commit`
- **Naming**: `mivia-small-<specialization>[-<variant>]`
- **Hardware & environment**: training runs on GPU when one is present;
  runtime inference is CPU-only under 2 GB RAM. The router is a 118 MB
  int8 ONNX encoder + 17 MB tokenizer at p50 ~8–16 ms / p95 ~12–21 ms on
  short text. `onnxruntime` + `tokenizers` are pinned runtime deps; ONNX
  artifacts are sha256-verified before load.

## Governing rules

1. **Rules first, ML where it helps.** Keep a grep/rules baseline as a
   measuring stick, not a gate.
2. **Constrained decoding for JSON validity.** The retry loop is a fallback
   (≤ 2 retries), not the primary mechanism.
3. **Accuracy over premature shipping.** The ≥ 99% held-out stop rule is a
   report line, not a stop sign. Expand training data and schemas whenever
   it helps.
4. **Persistence via skops.io only.** Pickle is banned (`.gitignore` blocks
   `*.pkl` / `*.pickle`).
5. **Version pinning required.** `pyproject.toml` is the single source of
   truth for pins; `requirements.txt` mirrors it (drift caught by
   `tests/test_smoke.py`), full freeze in `requirements-lock.txt`.
6. **No third-party dataset is committed.** This repository redistributes
   no upstream data — see [docs/DATA.md](docs/DATA.md). Builders fetch and
   construct corpora on demand; never paste fetched rows into the tree.
7. **Schemas are additive.** Register new schemas freely with
   `register_schema`; adding schemas re-freezes the fingerprint in
   `data/heldout/manifest.json` and regenerates the golden fixture — do
   both, and never leave the gates red.
8. **Golden capture pins behavior.** Regenerate the fixture when behavior
   legitimately changes; never hand-edit it.
9. **GPU training when a GPU is present.** Any training, fine-tuning, or
   hyperparameter search MUST use the GPU when one is available. GPU is a
   training-time resource only: the shipped artifact stays CPU-inferable.
10. **Instruments name the bytes they scored.** Every eval result records
    the sha256 of the candidate AND of the shipped bytes it measured.
    After a promotion, "candidate == shipped" proves nothing.
11. **A "drift" claim needs a reproduction attempt first.** Never register
    an instrument-drift adjudication without re-running the OLD model bytes
    through the CURRENT chain.
12. **Registered gates act, never just print.** A gate that only logs is
    vacuous — it must DROP the failing rows or FAIL the run. Corollaries:
    label files join to rows positionally and are verified by count;
    builders' provenance sections are idempotent.
13. **The pre-training audit gate MEASURES, never derives.** Mix totals and
    text-share come from measuring the actual base mix and the exact files
    the trainer will load — never from a recorded meta. The trainer refuses
    to start without the audit stamp.
14. **Focused tests in the dev loop; full suite once.** Run the focused
    test files for the surfaces you touched — the full suite is a
    pre-commit event only.
15. **Registered records cross-check their numbers.** Every number quoted
    in a result or training record is read back from the committed
    instrument before it is written. Include the failed lines of a cited
    record too — quoting only the green lines reads as cherry-picking.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # exact pins; full freeze in requirements-lock.txt
pip install -e .                  # src-layout package; dev extra: pip install -e ".[dev]"
python scripts/assemble_models.py # reassemble the split ONNX artifacts from models/parts/
pip install -e ".[training]"      # training extra (torch, transformers): GPU fine-tuning only
```

## Commands

| Purpose | Command |
|---|---|
| Tests | `pytest -q` |
| Generate synthetic data | `python -m mivia_small_json.synth` |
| Train the router (skops fallback) | `python -m mivia_small_json.train` |
| Fine-tune + export the ONNX router | `python scripts/export_encoder_onnx.py` |
| Build the guard's pair corpus | `python scripts/build_tool_match_corpus.py` |
| Fine-tune the tool-match guard | `python scripts/train_tool_match.py` |
| Measure a tool-match artifact | `python scripts/bench_tool_abstention.py` |
| Train the learned extractors | `python scripts/train_level_extractor.py` / `train_component_extractor.py` / `train_implicit_severity.py` |
| Benchmark vs rules baseline | `python -m mivia_small_json.bench` |
| LogHub benchmark | `python scripts/bench_loghub.py` |
| SWE-bench benchmark | `python scripts/bench_swebench.py` |
| Guard diagnosis | `python scripts/diagnose_tool_match.py` |
| Pre-training data audit gate | `python scripts/audit_training_mix.py` |
| Real-world slices benchmark | `python scripts/bench_realworld_slices.py` |
| Improvement cycle | `python -m mivia_small_json.flywheel --train` |
| Serve the model (HTTP + MCP) | `python -m mivia_small_json.api --host 0.0.0.0 --port 8000` |
| Refresh manifests | `python scripts/refresh_shipped_manifest.py` · `check_dataset_manifest.py --write` · `refresh-test-inventory.py` |

## Layout

```
docs/                 public documentation: USAGE, USE_CASES, MODEL-STATE,
                      LIMITATIONS, DATA, TOOL_ROUTING_AND_CLARIFICATION
src/mivia_small_json/ package: schemas, registry, router, guard, extractors,
                      pipeline, serving surfaces
data/synthetic/       committed synthetic corpora (seeded, small)
data/heldout/         frozen held-out splits — never trained on
models/               shipped artifacts + sha256 sidecars + SHIPPED.json;
                      oversized ONNX files are gitignored and ship as
                      models/parts/ byte-splits + manifest
scripts/              data builders (fetch-on-demand), benchmarks, training
tests/                the suite; fixtures under tests/fixtures/
```

## Definition of done (every change)

- Focused test files green in the loop; `pytest -q` green once before
  commit, and CI green on the PR.
- When a bench is re-run: the record carries all six metrics and the
  grep/rules baseline comparison, plus the environment fingerprint
  (python / scikit-learn / skops / numpy versions + git SHA).
- No unpinned dependency, no pickle, no third-party data in a commit.
- When an artifact changes: the before/after measurement is written down
  and the calibration re-checked.

## Router promotion checklist (every promotion, in order)

1. Verify the staged candidate's sha256 sidecars, then copy the int8 onnx +
   classes + tokenizer into `models/` (staging never writes there
   directly).
2. Write `models/router-encoder-meta.json` — it MUST keep
   `"in_domain_energy": -100.0` (the gate-off sentinel; the loader contract
   depends on it).
3. Refresh the `models/*.sha256` sidecars, then run
   `scripts/refresh_shipped_manifest.py`.
4. Re-split `models/parts/` — the oversized artifacts are GITIGNORED; a
   skipped re-split leaves a clean clone assembling the PREVIOUS bytes
   while `SHIPPED.json` claims the new ones.
5. Regenerate the golden fixture
   (`python tests/fixtures/make_routing_golden.py`); never hand-edit it.
6. Run the focused gates on the promoted bytes (golden, real-world bars,
   the guard and quality gates, serving tests).
7. Re-measure the external benchmarks on the FINAL bytes — numbers measured
   on the predecessor's bytes are stale the moment the bytes change.
8. Update `docs/MODEL-STATE.md` and the README numbers in the same commit.

## Data preparation protocol (before ANY training run)

1. **Composition is declared first, then audited.** Per-source /
   per-language / per-class row targets are declared BEFORE the corpus is
   built; the audit gate compares against the declaration, and a mismatch
   is a conscious re-registration, never an accident.
2. **Duplication is checked three ways** — within the corpus
   (strong-normalized, ≤ 1%), across sources (structural overlap dropped
   keep-first and recorded), and against the existing training mix
   (overlapping rows are hidden weight, not new signal).
3. **Overlap with eval is excluded by replication, not intent.** When a
   training source and an eval surface share an upstream file, the builder
   replicates the eval's own window logic and excludes exactly that window,
   verified by normalized-set comparison.
4. **Volume is verified, never assumed.** Every source has a declared
   minimum contribution; a builder that writes fewer rows than declared
   FAILS.
5. **Labels are adjudicated or mechanical — never vibes.** Teacher-authored
   data gets blind double adjudication with a kappa gate and a REAL drop
   path. Generated records never fabricate values: absent stays absent.
6. **Weakness is probed in the data BEFORE the GPU runs.** Route the new
   rows through the SHIPPED router first
   (`scripts/check_corpus_strength.py`): rows it already gets right are
   easy mass; rows it gets wrong are the actual signal.
7. **Balance bounds are declared per round and measured** from the files
   the trainer will load.
8. **Templates are capped**; real data replaces templates whenever supply
   exists.

## Training arms

- Register the round in a written spec BEFORE any candidate measurement:
  data plan, declared audit targets/bounds, bars, and the FAIL consequence.
  Never retune bars to a number you have already seen.
- Run the data preparation protocol above; the trainer refuses without the
  audit stamp. Same base checkpoint/recipe/seed unless the round explicitly
  registers a change.
- After a one-shot miss, dump the residual misses (language, routed_by,
  confidence, texture) and register the next increment to target THAT
  texture — not "more of the same".

## Guardrails

- Add a dependency only with an exact pin and a one-line justification.
- Keep `data/heldout/` for measurement only; do not train on it.
- Never set a weak-label threshold, a label-admission decision, or a cycle
  gate from eval data; the one exception is the flywheel's final held-out
  veto, a one-shot stop check that never feeds thresholds or tuning.
  Measure rule trust with a complete confusion matrix over labeled
  development sets. A class with too little support has unknown trust, and
  unknown trust never admits a label. Model probabilities are uncalibrated
  until a development calibration is supplied.
- Mechanical gates are deterministic tests only (sizes, checksums, validity
  rates) — no timing asserts; every gate must run inside a 2 GB CPU
  sandbox.
- Output formats have one source of truth:
  `src/mivia_small_json/schemas.py`; never duplicate field/enum lists
  elsewhere.
