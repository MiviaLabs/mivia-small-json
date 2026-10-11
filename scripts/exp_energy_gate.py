"""Retune the energy OOD gate on a larger, more diverse OOD set.

Phase 0 fit ``IN_DOMAIN_ENERGY`` on 18 OOD rows and saw in-domain fast-path
retention fall to 65.9% (max-softmax confidence >= 0.85 and energy >= 8.5).
This script rebuilds the measurement on the full nine ``data/real`` dev
splits (the natural in-domain distribution) and the committed OOD set
(``data/incoming/ood/ood-set.jsonl``), then sweeps the threshold.

Metrics, per threshold ``t``:
- **retention (conditioned)** = P(energy >= t | max-softmax >= 0.85): the
  fraction of rows the pre-gate fast path would have taken that the gate
  still lets through. This is the retention the gate controls; rows below
  0.85 confidence never fast-pathed and are unaffected.
- **retention (absolute)** = P(conf >= 0.85 and energy >= t) over all
  in-domain rows: the absolute fast-path rate. Its ceiling is the router's
  own confident rate, so it cannot reach 1.0.
- **OOD rejection** = P(energy < t).

Threshold-free: OOD AUROC and FPR@95TPR for the pre-gate score (max-softmax)
and the gate score (energy), computed with ``sklearn.metrics``.

The chosen threshold is the largest ``t`` with conditioned retention >= 0.90
that also keeps the confident-invoice anchor (energy 8.18) out of domain,
rounded to one decimal. A balanced per-class sample (100 rows/split) is
reported as a sensitivity, because the retention cost is concentrated in the
``commit`` and ``no_tool_call`` classes.

Writes ``data/incoming/ood/energy-sweep.json``. CPU-only, one forward pass
per row (batch 1, matching the serving path).
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path

from sklearn.metrics import roc_auc_score, roc_curve

from mivia_small_json.encoder import load_encoder

ROOT = Path(__file__).resolve().parents[1]
OOD_SET = ROOT / "data" / "incoming" / "ood" / "ood-set.jsonl"
REAL = ROOT / "data" / "real"
OUT = ROOT / "data" / "incoming" / "ood" / "energy-sweep.json"

SEED = 20261006
RETENTION_FLOOR = 0.90
# The confident-invoice anchor from tests/test_ood_fast_path.py; a threshold
# at or below its energy lets the >0.85 invoice fast-path to ``issue``.
INVOICE_LONG_ENERGY = 8.1765
SYSLOG = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
# The nine router-relevant dev splits (grade is a separate severity family).
IN_DOMAIN_SPLITS = ["agent_step", "commit", "diagnostic", "diff", "issue",
                    "log_event", "no_tool_call", "review_comment", "tool_call"]

# The dev splits are read from a pinned revision (default HEAD) so the
# measurement is reproducible even while the working tree is being edited by
# other work (the Phase 1 data expansion rewrites ``data/real``).
REV = "HEAD"


def _read_split(split: str) -> list[str]:
    rel = f"data/real/{split}/dev.jsonl"
    text_blob: str | None = None
    try:
        proc = subprocess.run(["git", "show", f"{REV}:{rel}"], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        text_blob = proc.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        text_blob = (ROOT / rel).read_text(encoding="utf-8")
    out: list[str] = []
    for line in text_blob.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("raw") or row.get("text") or ""
        if text.strip():
            out.append(text)
    return out


def pooled_rows() -> list[str]:
    """All nine dev splits, natural weighting (the in-domain distribution)."""
    out: list[str] = []
    for split in IN_DOMAIN_SPLITS:
        out.extend(_read_split(split))
    return out


def balanced_rows(per_split: int = 100) -> list[str]:
    rng = random.Random(SEED)
    out: list[str] = []
    for split in IN_DOMAIN_SPLITS:
        rows = _read_split(split)
        rng.shuffle(rows)
        out.extend(rows[:per_split])
    return out


def ood_rows() -> list[str]:
    return [json.loads(line)["text"] for line in
            OOD_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def scores(enc, texts: list[str]) -> tuple[list[float], list[float]]:
    conf: list[float] = []
    energy: list[float] = []
    for text in texts:  # batch 1: matches the per-row serving path
        probs, eng = enc.predict_proba_with_energy([text])
        conf.append(float(probs[0].max()))
        energy.append(float(eng[0]))
    return conf, energy


def fpr_at_95tpr(labels: list[int], s: list[float]) -> float:
    """FPR at the threshold where TPR first reaches 95% (label 1 = in-domain)."""
    fpr, tpr, _ = roc_curve(labels, s)
    for f, t in zip(fpr, tpr):
        if t >= 0.95:
            return float(f)
    return 1.0


def curve(pre_gate: list[float], all_energy: list[float],
          ood_energy: list[float], n_in: int) -> list[dict]:
    rows = []
    for t in [round(x * 0.25, 2) for x in range(20, 44)]:  # 5.00 .. 10.75
        cond = sum(e >= t for e in pre_gate) / len(pre_gate)
        absolute = sum(e >= t for e in all_energy) / n_in
        rej = sum(e < t for e in ood_energy) / len(ood_energy)
        rows.append({"threshold": t, "retention_conditioned": round(cond, 4),
                     "retention_absolute": round(absolute, 4),
                     "ood_rejection": round(rej, 4)})
    return rows


def pick(pre_gate: list[float], floor: float,
         min_threshold: float) -> float:
    """Largest 0.1 threshold >= floor retention and > the invoice energy."""
    chosen = None
    for t in sorted({round(e, 1) for e in pre_gate}):
        if t <= min_threshold:
            continue
        if sum(e >= t for e in pre_gate) / len(pre_gate) >= floor:
            chosen = t
    if chosen is None:  # floor unreachable above the invoice: best we can do
        chosen = round(min(e for e in pre_gate if e > min_threshold), 1)
    return chosen


def main() -> int:
    global REV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev", default="HEAD",
                    help="git revision to read data/real dev splits from")
    args = ap.parse_args()
    REV = args.rev

    enc = load_encoder()
    if enc is None:
        print("shipped ONNX encoder absent; nothing to sweep")
        return 1

    ind = pooled_rows()
    bal = balanced_rows()
    ood = ood_rows()
    print(f"pooled in-domain {len(ind)} · balanced in-domain {len(bal)} · "
          f"OOD {len(ood)}")

    ind_conf, ind_energy = scores(enc, ind)
    bal_conf, bal_energy = scores(enc, bal)
    ood_conf, ood_energy = scores(enc, ood)

    def pre_gate(conf, energy):
        return [e for c, e in zip(conf, energy) if c >= 0.85]

    pg_pool = pre_gate(ind_conf, ind_energy)
    pg_bal = pre_gate(bal_conf, bal_energy)

    # Threshold-free metrics on the pooled set.
    labels = [1] * len(ind) + [0] * len(ood)
    auroc_soft = roc_auc_score(labels, ind_conf + ood_conf)
    auroc_energy = roc_auc_score(labels, ind_energy + ood_energy)
    fpr_soft = fpr_at_95tpr(labels, ind_conf + ood_conf)
    fpr_energy = fpr_at_95tpr(labels, ind_energy + ood_energy)
    # Same metrics on the balanced sample, for a Phase-0-comparable reading.
    labels_bal = [1] * len(bal) + [0] * len(ood)
    auroc_soft_bal = roc_auc_score(labels_bal, bal_conf + ood_conf)
    auroc_energy_bal = roc_auc_score(labels_bal, bal_energy + ood_energy)
    fpr_soft_bal = fpr_at_95tpr(labels_bal, bal_conf + ood_conf)
    fpr_energy_bal = fpr_at_95tpr(labels_bal, bal_energy + ood_energy)

    pool_curve = curve(pg_pool, ind_energy, ood_energy, len(ind))
    bal_curve = curve(pg_bal, bal_energy, ood_energy, len(bal))

    chosen = pick(pg_pool, RETENTION_FLOOR, INVOICE_LONG_ENERGY)
    chosen_cond = sum(e >= chosen for e in pg_pool) / len(pg_pool)
    chosen_abs = sum(e >= chosen for e in ind_energy) / len(ind)
    chosen_rej = sum(e < chosen for e in ood_energy) / len(ood_energy)
    chosen_bal = sum(e >= chosen for e in pg_bal) / len(pg_bal)

    youden = max([r["threshold"] for r in pool_curve],
                 key=lambda t: (sum(e >= t for e in pg_pool) / len(pg_pool)
                                - sum(e >= t for e in ood_energy)
                                / len(ood_energy)))

    result = {
        "data_rev": REV,
        "n_in_domain_pooled": len(ind),
        "n_in_domain_balanced": len(bal),
        "n_ood": len(ood),
        "pre_gate_pooled": len(pg_pool),
        "pre_gate_balanced": len(pg_bal),
        "auroc_max_softmax": round(float(auroc_soft), 4),
        "auroc_energy": round(float(auroc_energy), 4),
        "fpr95_max_softmax": round(float(fpr_soft), 4),
        "fpr95_energy": round(float(fpr_energy), 4),
        "auroc_max_softmax_balanced": round(float(auroc_soft_bal), 4),
        "auroc_energy_balanced": round(float(auroc_energy_bal), 4),
        "fpr95_max_softmax_balanced": round(float(fpr_soft_bal), 4),
        "fpr95_energy_balanced": round(float(fpr_energy_bal), 4),
        "retention_floor": RETENTION_FLOOR,
        "chosen_threshold": chosen,
        "chosen_retention_conditioned": round(chosen_cond, 4),
        "chosen_retention_absolute": round(chosen_abs, 4),
        "chosen_retention_balanced": round(chosen_bal, 4),
        "chosen_ood_rejection": round(chosen_rej, 4),
        "youden_threshold": youden,
        "invoice_long_energy": INVOICE_LONG_ENERGY,
        "syslog_energy": round(float(enc.in_domain_scores([SYSLOG])[0]), 4),
        "curve_pooled": pool_curve,
        "curve_balanced": bal_curve,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"AUROC max-softmax {auroc_soft:.4f} -> energy {auroc_energy:.4f}")
    print(f"FPR@95TPR max-softmax {fpr_soft:.4f} -> energy {fpr_energy:.4f}")
    print(f"  balanced: AUROC {auroc_soft_bal:.4f} -> {auroc_energy_bal:.4f} · "
          f"FPR@95TPR {fpr_soft_bal:.4f} -> {fpr_energy_bal:.4f}")
    print(f"syslog {result['syslog_energy']} · invoice_long "
          f"{INVOICE_LONG_ENERGY} · chosen {chosen}")
    print(f"chosen retention pooled {chosen_cond:.3f} (abs {chosen_abs:.3f}) "
          f"· balanced {chosen_bal:.3f} · OOD rejection {chosen_rej:.3f}")
    print("\nthreshold  ret_pool  ret_bal  ret_abs  ood_rej")
    for pr, br in zip(pool_curve, bal_curve):
        print(f"{pr['threshold']:>9.2f}  {pr['retention_conditioned']:>8.3f}  "
              f"{br['retention_conditioned']:>7.3f}  "
              f"{pr['retention_absolute']:>7.3f}  {pr['ood_rejection']:>7.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
