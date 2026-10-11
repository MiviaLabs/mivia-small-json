#!/usr/bin/env python3
"""Calibration of the ACTIVE ONNX encoder router on the dev reference.

`data/calibration.json` measures the skops fallback (sklearn pipeline,
`label.measure_dev_calibration`). This script measures the shipped
production router (`models/router-encoder-int8.onnx` via
`pipeline.load_active_router`) with the same statistics — ECE, worst band,
Brier — over `data/heldout/dev-reference.jsonl`, so the production path
has a committed calibration record too (gap G4,
reality-evals §2).

Mirrors the skops measurement's headline fields so the two artifacts are
directly comparable. Read-only; writes
`reports/router-calibration-onnx.json`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.pipeline import load_active_router  # noqa: E402

DEV = ROOT / "data" / "heldout" / "dev-reference.jsonl"
OUT = ROOT / "reports" / "router-calibration-onnx.json"
BANDS = 10
THRESHOLD = 0.9


def main() -> int:
    router = load_active_router()
    if router is None:
        raise SystemExit("active ONNX router not loadable")
    rows = [json.loads(line) for line in DEV.read_text().splitlines()
            if line.strip()]
    correct = 0
    total = 0
    brier = 0.0
    band_stats: dict[int, dict] = {}
    for row in rows:
        raw, gold = row["raw"], row["schema"]
        proba = router.predict_proba([raw])[0]
        classes = list(router.classes)
        pairs = sorted(zip(classes, proba), key=lambda t: -t[1])
        pred, conf = pairs[0]
        conf = float(conf)
        hit = int(pred == gold)
        correct += hit
        total += 1
        p_true = float(dict(zip(classes, proba)).get(gold, 0.0))
        pred = str(pred)
        brier += (1.0 - p_true) ** 2
        band = min(int(conf * BANDS), BANDS - 1)
        st = band_stats.setdefault(band, {"n": 0, "correct": 0, "conf": 0.0})
        st["n"] += 1
        st["correct"] += hit
        st["conf"] += conf

    ece = 0.0
    worst = 0.0
    worst_band = None
    bands = []
    for band in sorted(band_stats):
        st = band_stats[band]
        acc = st["correct"] / st["n"]
        conf = st["conf"] / st["n"]
        err = abs(acc - conf)
        ece += st["n"] / total * err
        bands.append({"band": band, "n": st["n"],
                      "confidence": round(conf, 4),
                      "accuracy": round(acc, 4),
                      "abs_error": round(err, 4)})
        if st["n"] >= 30 and err > worst:
            worst, worst_band = err, band
    payload = {
        "router": "models/router-encoder-int8.onnx (active ONNX encoder)",
        "measured_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dev_reference": "data/heldout/dev-reference.jsonl",
        "n": total,
        "accuracy": round(correct / total, 4),
        "brier": round(brier / total, 4),
        "ece": round(ece, 4),
        "max_abs_error": round(worst, 4),
        "max_abs_error_band": worst_band,
        "bands": bands,
        "note": ("mirrors data/calibration.json's headline fields for the "
                 "skops fallback; skops ECE 0.0103 on the same reference"),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "bands"},
                     indent=1))
    for b in bands:
        print(f"  band {b['band']}: n={b['n']:>5} conf {b['confidence']:.3f} "
              f"acc {b['accuracy']:.3f} |err| {b['abs_error']:.3f}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
