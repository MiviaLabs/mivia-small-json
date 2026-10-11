#!/usr/bin/env python3
"""Phase-1 mix admission: a pre-registered, task-matched OOD bar.

Admits the Phase-1 GitBugs + CommitChronicle train rows (``curate_data_phase1``
train-only, ``DEV_PCT = 0``) and retrains both routers, then measures the
retrained ONNX router against the shipped one on a held-out drawn from the
**same distribution** as the new rows:

- issue: NLBSE'24 test split (``data/incoming/external/nlbse24/issues.jsonl``).
- commit: the project/sha-disjoint CommitChronicle held-out
  (``data/incoming/external/phase1-ood/commit.jsonl``).

Both are hash-disjoint from every admitted train row.

PRE-REGISTERED BAR (stated 2026-10-06, before any measurement). PASS iff, for
the retrained ONNX router versus the shipped ONNX router:

1. new held-out accuracy (issue + commit pooled) improves by >= 1.0 pp, AND
2. the frozen 4-schema held-out (``data/heldout/test.jsonl``) routed-and-valid
   stays >= 99.0%, AND
3. no existing OOD set (nlbse / cleancodereview / bfcl / no_tool_call)
   regresses by more than 1.0 pp.

Otherwise REJECT. GPU for training (stated), CPU for inference.

Writes ``reports/exp-phase1-admission.md``. Stdlib + the training extra.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

# --- PRE-REGISTERED BAR (do not tune after seeing the numbers) ----------------
HELDOUT_GAIN_PP = 1.0      # (1) improvement on the new held-out
HELDOUT_FLOOR = 99.0       # (2) frozen 4-schema held-out routed-and-valid
OOD_MAX_REGRESSION_PP = 1.0  # (3) allowed regression per existing OOD set

NEW_HOLDOUT = (
    ("issue", ROOT / "data/incoming/external/nlbse24/issues.jsonl", "issue"),
    ("commit", ROOT / "data/incoming/external/phase1-ood/commit.jsonl", "commit"),
)
FROZEN_HELDOUT = ROOT / "data/heldout/test.jsonl"
OOD = (
    ("nlbse", ROOT / "data/incoming/external/nlbse/issues.jsonl", "issue"),
    ("cleancodereview",
     ROOT / "data/incoming/external/cleancodereview/comments.jsonl",
     "review_comment"),
    ("bfcl", ROOT / "data/incoming/external/bfcl/calls.jsonl", "tool_call"),
    ("no_tool_call", ROOT / "data/real/no_tool_call/dev.jsonl", "no_tool_call"),
)
OOD_CAP = 400
STAGING = ROOT / "data" / "incoming" / "models"
REPORT = ROOT / "reports" / "exp-phase1-admission.md"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .split("\n") if line.strip()]


def _register_schemas() -> None:
    from mivia_small_json import bootstrap
    from mivia_small_json.agent_step import register_agent_step_schema
    from mivia_small_json.diagnostic import register_diagnostic_schema
    from mivia_small_json.diff_comment import register_diff_and_review_schemas
    from mivia_small_json.tool_call import (register_no_tool_call_schema,
                                            register_tool_call_schema)

    bootstrap.register_example_schemas()
    register_tool_call_schema()
    register_no_tool_call_schema()
    register_diagnostic_schema()
    register_diff_and_review_schemas()
    register_agent_step_schema()


def _write_sidecar(path: Path) -> None:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / (path.name + ".sha256")).write_text(digest + "\n",
                                                       encoding="utf-8")


def _load_candidate():
    """Load the freshly exported int8 ONNX router (sha-verified)."""
    from mivia_small_json import encoder

    onnx = STAGING / "router-encoder-int8.onnx"
    tok = STAGING / "minilm" / "tokenizer.json"
    classes = json.loads((STAGING / "router-encoder-classes.json")
                         .read_text(encoding="utf-8"))
    _write_sidecar(onnx)
    _write_sidecar(tok)
    return encoder.load_encoder(path=onnx, tokenizer_path=tok,
                                classes=tuple(classes))


def _new_heldout(router) -> tuple[float, dict]:
    """Pooled routed-to-expected accuracy on the new held-out (issue+commit)."""
    from mivia_small_json.pipeline import classify

    per: dict[str, tuple[int, int]] = {}
    ok = total = 0
    for name, path, expected in NEW_HOLDOUT:
        rows = _rows(path)
        hit = sum(1 for r in rows
                  if classify(r.get("text") or r["raw"], router)[0]
                  .get("type") == expected)
        per[name] = (hit, len(rows))
        ok += hit
        total += len(rows)
    return (100 * ok / total if total else float("nan")), per


def _frozen_heldout(router) -> float:
    """Frozen 4-schema held-out: routed-and-valid %."""
    from mivia_small_json.pipeline import classify

    rows = _rows(FROZEN_HELDOUT)
    ok = sum(1 for r in rows
             if (lambda o, m: m["valid"] and o.get("type") == r["schema"])(
                 *classify(r["raw"], router)))
    return 100 * ok / len(rows)


def _ood(router) -> dict[str, float]:
    from mivia_small_json.pipeline import classify

    out: dict[str, float] = {}
    for name, path, expected in OOD:
        rows = _rows(path)[:OOD_CAP]
        hit = sum(1 for r in rows
                  if classify(r.get("text") or r["raw"], router)[0]
                  .get("type") == expected)
        out[name] = 100 * hit / len(rows) if rows else float("nan")
    return out


def main() -> int:
    import export_encoder_onnx as exp  # noqa: E402  (scripts/ on sys.path)

    _register_schemas()
    from mivia_small_json import encoder, model

    seed_everything(42)
    shipped = encoder.load_encoder()
    if shipped is None:
        print("shipped ONNX router unavailable; cannot compare", file=sys.stderr)
        return 2

    # --- retrain (GPU) -------------------------------------------------------
    print("== retraining the ONNX encoder on the admitted mix ==")
    net, tok, classes = exp.train()
    exp.export_onnx(net, tok, classes)
    candidate = _load_candidate()
    if candidate is None:
        print("candidate ONNX export failed to load", file=sys.stderr)
        return 2

    # skops retrained to a staging path (the shipped skops + its calibration
    # stay untouched; the ONNX is the active router).
    print("== retraining the skops router (staging) ==")
    skops = model.main(root=ROOT,
                       out_path=STAGING / "router-phase1.skops")

    # --- measure -------------------------------------------------------------
    before_new, before_per = _new_heldout(shipped)
    after_new, after_per = _new_heldout(candidate)
    before_frozen = _frozen_heldout(shipped)
    after_frozen = _frozen_heldout(candidate)
    before_ood = _ood(shipped)
    after_ood = _ood(candidate)

    gain = after_new - before_new
    bar1 = gain >= HELDOUT_GAIN_PP
    bar2 = after_frozen >= HELDOUT_FLOOR
    regressions = {name: before_ood[name] - after_ood[name] for name in before_ood}
    bar3 = all(r <= OOD_MAX_REGRESSION_PP for r in regressions.values())
    passed = bar1 and bar2 and bar3

    def _fmt(x: float) -> str:
        return "n/a" if x != x else f"{x:.1f}"

    lines = [
        "# Phase-1 mix admission — pre-registered, task-matched OOD bar",
        "",
        "2026-10-06 · branch `dev` · retrained on the admitted mix "
        "(issue 16861 → 31861, commit 3095 → 9095; train-only, `DEV_PCT = 0`).",
        "",
        "## Pre-registered bar (stated before measuring)",
        "",
        "**PASS** iff, for the retrained ONNX router vs the shipped ONNX router:",
        "",
        f"1. new held-out accuracy (issue + commit pooled) improves by "
        f"≥ {HELDOUT_GAIN_PP:.1f} pp;",
        f"2. frozen 4-schema held-out routed-and-valid stays ≥ "
        f"{HELDOUT_FLOOR:.0f}%;",
        f"3. no existing OOD set regresses by more than "
        f"{OOD_MAX_REGRESSION_PP:.1f} pp.",
        "",
        "## The held-out",
        "",
        "Drawn from the same distribution as the new rows, disjoint from every",
        "admitted train row (0 text, 0 sha, 0 project overlap):",
        "",
        f"- issue: NLBSE'24 test split — {len(_rows(NEW_HOLDOUT[0][1]))} rows;",
        f"- commit: project/sha-disjoint CommitChronicle held-out — "
        f"{len(_rows(NEW_HOLDOUT[1][1]))} rows.",
        "",
        "## Before / after",
        "",
        "| metric | shipped | retrained | Δ |",
        "|---|---|---|---|",
        f"| **new held-out** (issue+commit) % | {_fmt(before_new)} | "
        f"{_fmt(after_new)} | **{gain:+.1f}** |",
    ]
    for name in ("issue", "commit"):
        b, n = before_per[name]
        a, n2 = after_per[name]
        lines.append(f"| · new held-out {name} % | {100 * b / n:.1f} | "
                     f"{100 * a / n2:.1f} | {100 * (a / n2 - b / n):+.1f} |")
    lines.append(f"| frozen 4-schema held-out % | {_fmt(before_frozen)} | "
                 f"{_fmt(after_frozen)} | {after_frozen - before_frozen:+.1f} |")
    for name in before_ood:
        lines.append(f"| OOD {name} % | {_fmt(before_ood[name])} | "
                     f"{_fmt(after_ood[name])} | "
                     f"{after_ood[name] - before_ood[name]:+.1f} |")
    lines += [
        f"| skops real-dev accuracy | — | {skops['real_dev_accuracy']:.4f} | — |",
        f"| skops real-dev macro F1 | — | {skops['real_dev_macro_f1']:.4f} | — |",
        "",
        "## Verdict",
        "",
        f"- bar 1 (held-out ≥ +{HELDOUT_GAIN_PP:.1f} pp): "
        f"**{'PASS' if bar1 else 'FAIL'}** ({gain:+.1f} pp)",
        f"- bar 2 (frozen held-out ≥ {HELDOUT_FLOOR:.0f}%): "
        f"**{'PASS' if bar2 else 'FAIL'}** ({after_frozen:.1f}%)",
        f"- bar 3 (no OOD regression > {OOD_MAX_REGRESSION_PP:.1f} pp): "
        f"**{'PASS' if bar3 else 'FAIL'}** "
        f"(max regression {max(regressions.values()):+.1f} pp)",
        "",
        f"**{'PASS — promote.' if passed else 'REJECT — revert the data and model artifacts.'}**",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.exit(main())
