# Limitations

What mivia-small-json cannot do — measured, named, and published next to
the wins. Every number below is a measurement on its named frozen set;
several are pinned as one-sided floors in the test suite.

## Routing and extraction

| Surface | Result | Note |
|---|---|---|
| SWE-bench gold patches → `diff` | 98.0% | 2 of 100 unified patches route elsewhere; output stays valid JSON, file-path extraction holds on the routed rows. A ±6pp run-variance band exists on this surface — quote it. |
| SWE-bench problem-statement titles | 7.2% exact | Long problem statements without a crisp title sentence defeat the title extractor; body recall stays ~0.996. |
| Real-world issues | 88.8% | The remainder is wrong-class on near-neighbor look-alike text; the output stays valid JSON either way. |
| Real PRs | 80.3% | Third-party PR text is the hard tail; validity does not drop with it. |
| German incident tickets | 98.7% | Near its structural ceiling. |
| Exact severity on real business tickets | 46.8% exact / 84.2% within-1 (QWK 0.675) | Exact-match inherits annotator label noise (kappa 0.717) — treat exact numbers as ceiling-limited. Middle tier 2 on ambiguous text is the weakest cell. |

## Zero-shot generalization

| Surface | Result | Note |
|---|---|---|
| S1MB — 106 alien task families (26,269 decisions) | 35.5% | Equal to the label-set random baseline (35.19%): the aggregate is at-random. Skill does not transfer to arbitrary label taxonomies without examples. |
| JevBench v1 — general decisions | 54.1% | Easy 85.4% / original 52.8% / hard 41.4% across the three splits. |
| Banking77 zero-shot select (77 intents, 600-row fresh reserve) | 81.0% top-1 | Dedicated zero-shot select-rank head, no task training; conformal confidence gating is the abstention channel, per-label example enrichment is a caller-side lift. The 85% trained-head-parity mark is published as not reached. |

## Guard

| Surface | Result | Note |
|---|---|---|
| Guard residual categories | infeasible 36.7% / sibling_swap 63.5% / ask_followup 74.8% | Needs architecture work, not data; the served outcome for unsure cases is the honest CLARIFY path. |
| String-parameter slot filling | 89.07% (163/183) on the BFCL calls set | Tools whose required parameters are all strings fill only what the request text verbatim carries; the rest abstain to a clarification instead of inventing values. |

## Serving envelope

- Long-input latency is load-dependent, not a model property: ~11k-character
  prose runs 63–87 ms on a quiet box; the encoder caps at 512 tokens by
  design, and input capping is deliberately not applied.
- Built into the server: bearer auth, a Host allow-list, opt-in per-key
  metering with an exact 429 quota wall, and byte-verified zero retention.
  TLS termination and OAuth stay reverse-proxy territory.
- Concurrent serving is bounded by the inference gate (default 8, queue-not-
  reject) to stay inside the 2 GB memory envelope on CPU.

## Retention (the model's own)

Ticket-class retention on the committed one-shot: 1.00 on non-English
tickets (150/150), 0.99 on German tickets — minimum retention 0.99 across
every held-out workflow surface. Routing OOD rejection 92.23%, free-chat
98.63%. The seven pre-registered real-world slice floors all PASS on the
shipped bytes; strict accuracy counts rows with confident gold labels, and
ambiguous rows are excluded, never hidden.
