# MODEL STATE

The measured state of the shipped model. Every number is a measurement on
its named frozen set; nothing here is derived or projected. Known limits
live next to the wins in [LIMITATIONS.md](LIMITATIONS.md); the data policy
is [DATA.md](DATA.md).

## Shipped artifacts

`models/SHIPPED.json` lists every shipped artifact with its sha256 sidecar.
The router and cross-encoder artifacts are int8 ONNX; the decision heads and
extractors are skops. Bases: `microsoft/Multilingual-MiniLM-L12-H384` (MIT)
and `cross-encoder/ms-marco-MiniLM-L-6-v2` / `-L-12-v2` (Apache-2.0) —
fine-tuned and quantized; see [NOTICE](NOTICE.md).

## Per-component state

| Component | Evaluated on | Result |
|---|---|---|
| Router (11 classes: 10 workflow + plain `text`) | frozen held-out | routed-and-valid **99.71%** (F1 0.9975, valid 100%, p95 11.7 ms) |
| Router — LogHub 32k (16 systems) | LogHub ground truth | routed 100.0% / valid 100% / component 99.99% / explicit level 99.98% |
| Router — SWE-bench | SWE-bench Lite subset | problems **97/100**, patches **98/100**, validity 100% (±6pp run-variance band on this surface — quote it) |
| Router — external hosted-router head-to-head | identical rows | **99.71 vs 97.14** |
| Router — non-workflow one-shot | registered bars | rejection **92.23%**, free-chat **98.63%**, retention min **0.99** (tickets-non-en 1.0, tickets-de 0.99) |
| Router — real-world slices | seven pre-registered floors (`tests/test_realworld_bars.py`) | chat **93.5** / fa-ar **98.8** / fa-fa **99.4** / issue **88.8** / long-log **100.0** / post-commit **100.0** / slack **95.3** — all floors PASS |
| Router — tool catalogues | public BFCL calls set (183) | **89.07%** routed-to-expected through the served path with deterministic verbatim string filling; misses abstain honestly |
| Component / level extractors | LogHub 32k | component **100.0%**, explicit level **99.98%** |
| Prose title/body extractor | reality samples (1k en + 1k de tickets, 300 PRs) | title exact **71.8 / 67.5%** en/de, body recall **0.997 / 0.995**; PRs 21.7% / 0.958 |
| Log templates | LogHub 16 systems | GA **0.8873** / PA 0.4063 |
| Tool-match guard | guard eval (5,100 rows) | **79.7% abstention @ 95.1% retention** (threshold 0.6020); catalogue AUC **0.9030**; no_tool_needed **95.4%**; session-transfer rank serves **71.5%** of real agent-session invocations (AUC 0.93) |
| Guard residuals | same eval | infeasible 36.7% / sibling_swap 63.5% / ask_followup 74.8% — the served outcome for unsure cases is the honest CLARIFY path |
| Zero-shot select (tool head) | Banking77 fresh reserve (n=600) | **81.0% top-1** with the wired select-rank head; binary referee expected-yes/no **99.6 / 99.6**; abstention via band-tau confidence gating; rollback `MIVIA_SELECT_RANK_HEAD=legacy` |
| Intent selection (`head="intent"`) | language-gated dual head | EN dev **81.75%**, multilingual dev **61.64%** through the dispatch |
| Severity — canonical register | register panel (n=217) | **70.0%** exact / 93.1% within-1 (head alone); production panel 67.7% / 93.5% / QWK 0.723 |
| Severity — real business register | v3 real tickets (n=600, primary) | 46.8% exact / 84.2% within-1 / QWK 0.675 (register gate → specialist); tiers {0: 78.4, 1: 21.5, 2: 9.6, 3: 98.0} |
| Severity — German | frozen de reserve (370) | within-1 **90.5%** on the de head panel; **90.3%** through triage composition (hashed-feature lang gate, first-load p50 5.8 ms, tau 0.4844) |
| Router calibration | frozen dev (n=2,800) | ECE **0.0017** |
| Load envelope | concurrent serving test | worst 200-tool cell 1,662 MB under the 2 GB cap; p50 562 ms sequential; inference gate default 8, queue-not-reject |
| is-security gate (candidate, not wired) | frozen 2,000-row slice | F1 **0.9985**; operational-language probe scores 0.209 < tau — recorded limitation |
| Security band head (candidate, FAIL — not shipped) | NVD temporal reserve (2,811) | exact 48.5% / within-1 91.7% / macro-F1 0.447 — NVD vendor band labels noisy at boundaries |

## Label-quality ceiling on the severity surfaces

v3 gold is agent-annotated with a published guideline and a blind re-check;
Cohen's kappa **0.717** (substantial), with all disagreements at the 1↔2 and
2↔3 boundaries. Absolute exact-match scores therefore carry that label
noise; deltas between systems measured on the same rows remain decisive.
Spot-review of v3 rows is the cheapest available quality upgrade.

## Measurement policy (binding)

1. Every quoted number names the set it was measured on, and that set is
   disjoint from the training data.
2. New eval slices get a reserve file, provenance, and a published
   inter-annotator kappa before use; no eval is edited after freezing.
3. Bars are pre-registered before training and never reinterpreted after
   numbers exist; a failed bar with a near-miss goes to the maintainer
   queue as a decision, not quietly re-run.
4. Minimum detectable effects at the working sizes: v3 ≈ ±8pp, panel ≈
   ±13pp (80% power, two-sided 5%).
5. Weak-label trust is never set from eval data; a class with too little
   development support has unknown trust, and unknown trust admits nothing.
