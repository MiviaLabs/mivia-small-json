#!/usr/bin/env python3
"""Energy-gate recalibration for the 10-class candidate router.

The phase-4.5 NO-GO lesson: a new head's logit geometry invalidates the
shipped IN_DOMAIN_ENERGY. This re-derives the threshold on the
CANDIDATE's own logits with the original protocol (exp_energy_gate.py):
the largest 0.1-step threshold that keeps pooled in-domain fast-path
retention (P(energy >= t | max-softmax >= 0.85)) >= 0.90 while the
confident-invoice anchor still falls through; the syslog anchor must
pass. Dev splits + hand-written anchors only — no eval surface feeds the
threshold. Writes the chosen threshold + curve to stdout as JSON.

    .venv/bin/python scripts/calibrate_energy_10class.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CAND = ROOT / "data/incoming/models/mm12-10class-s42"
STEM = "mm12-10class-s42"

INVOICE = ("Invoice #INV-2026-991: Payment of $450 for cloud consulting "
           "services due on 2026-11-01")
# the calibration anchor is the LONG invoice (exp_energy_gate's 8.18 anchor)
INVOICE_LONG = (
    "INVOICE\nInvoice Number: INV-2026-0042\nBilling Date: 2026-03-01\n"
    "Due Date: 2026-04-01\nCustomer: Acme Corporation\nLine Items:\n"
    "  Cloud consulting services - 40 hours @ $150/hr = $6,000.00\n"
    "  Software license renewal - $1,200.00\nSubtotal: $7,200.00\n"
    "Tax (8.25%): $594.00\nTotal Due: $7,794.00\n"
    "Please remit payment within 30 days.\n")
SYSLOG = "2026-03-01T12:00:00Z ERROR [auth]: token expired"
# The nine router-relevant dev splits (grade is a separate severity family).
IN_DOMAIN_SPLITS = ["agent_step", "commit", "diagnostic", "diff", "issue",
                    "log_event", "no_tool_call", "review_comment", "tool_call"]
FLOOR = 0.90


def main() -> int:
    import numpy as np
    from tokenizers import Tokenizer

    import onnxruntime as ort
    from mivia_small_json import integrity
    from mivia_small_json.encoder import HEAD_LEN, MAX_LEN, TAIL_LEN
    from exp_energy_gate import pooled_rows  # scripts/ is on sys.path

    onnx_p = CAND / f"{STEM}-int8.onnx"
    tok_p = CAND / f"{STEM}-tokenizer.json"
    for p in (onnx_p, tok_p):
        if not integrity.verified(p):
            print(f"candidate artifact failed sha check: {p}", file=sys.stderr)
            return 2
    sess = ort.InferenceSession(str(onnx_p),
                                providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(tok_p))
    # the converted tokenizer.json bakes truncation AND padding to 16 —
    # the served loader disables both (encoder.py: no_truncation+no_padding)
    tok.no_truncation()
    tok.no_padding()

    def score(texts: list[str]) -> tuple[list[float], list[float]]:
        conf, energy = [], []
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]
            encs = [tok.encode(t) for t in chunk]
            # head-tail truncation to MAX_LEN, exactly as served
            ids = []
            for e in encs:
                if len(e.ids) > MAX_LEN:
                    head = e.ids[1:1 + HEAD_LEN]
                    tail = e.ids[-(TAIL_LEN + 1):-1]
                    ids.append([e.ids[0]] + head + tail + [e.ids[-1]])
                else:
                    ids.append(e.ids)
            pad = max(len(x) for x in ids)
            pad_id = tok.token_to_id("<pad>") or 0
            ii = np.full((len(ids), pad), pad_id, dtype=np.int64)
            am = np.zeros((len(ids), pad), dtype=np.int64)
            for j, x in enumerate(ids):
                ii[j, :len(x)] = x
                am[j, :len(x)] = 1
            logits = sess.run(None, {"input_ids": ii,
                                     "attention_mask": am})[0]
            mx = logits.max(-1, keepdims=True)
            e = np.exp(logits - mx)
            probs = e / e.sum(-1, keepdims=True)
            conf.extend(probs.max(-1).tolist())
            energy.extend((mx[:, 0] + np.log(e.sum(-1))).tolist())
        return conf, energy

    rows = pooled_rows()
    conf, energy = score(rows)
    pre_gate = [e for c, e in zip(conf, energy) if c >= 0.85]
    print(f"pooled dev rows={len(rows)} pre-gate(conf>=0.85)={len(pre_gate)}")

    inv_conf, inv_energy = score([INVOICE, INVOICE_LONG])
    sys_conf, sys_energy = score([SYSLOG])
    inv_e, sys_e = inv_energy[1], sys_energy[0]
    print(f"anchors: invoice_long energy={inv_e:.2f} conf={inv_conf[1]:.3f} · "
          f"invoice_short energy={inv_energy[0]:.2f} conf={inv_conf[0]:.3f} · "
          f"syslog energy={sys_e:.2f} conf={sys_conf[0]:.3f}")

    chosen = None
    curve = []
    for t in [round(x * 0.1, 1) for x in range(50, 110)]:  # 5.0 .. 10.9
        ret = sum(e >= t for e in pre_gate) / len(pre_gate)
        curve.append({"threshold": t, "retention_conditioned": round(ret, 4)})
        if ret >= FLOOR and t > inv_e:
            chosen = t
    if chosen is None:
        print("NO threshold satisfies both clauses on the candidate "
              "geometry", file=sys.stderr)
        return 2
    syslog_pass = sys_e >= chosen
    out = {
        "candidate": onnx_p.name,
        "chosen_IN_DOMAIN_ENERGY": chosen,
        "retention_at_chosen": next(c["retention_conditioned"]
                                    for c in curve
                                    if c["threshold"] == chosen),
        "invoice_long_energy": round(inv_e, 4),
        "syslog_energy": round(sys_e, 4),
        "syslog_passes": syslog_pass,
        "curve": curve,
    }
    if not syslog_pass:
        print("syslog anchor FAILS the chosen threshold", file=sys.stderr)
        return 2
    (CAND / "energy-calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
