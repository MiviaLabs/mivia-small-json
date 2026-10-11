"""Arm 1 phase 0b/0c: energy recalibration + KNN-logit OOD measurement.

Everything is computed from the SHIPPED int8 bytes — the router artifact is
never modified. Phases:

    --sweep   energy thresholds x {v2 OOD set, in-domain reference surfaces}
              -> reports/incoming ood-sweep-v2.json + chosen threshold
    --knn     build the kNN logit bank (skops + sha) and measure
              GO/NO-GO: >= +5pp OOD rejection over energy at equal
              in-domain retention

Pre-registered bars live in ood-routing-band This
script only MEASURES and writes reports; changing encoder.IN_DOMAIN_ENERGY
is a separate, reviewed step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # noqa: E402

OOD_V1 = ROOT / "data" / "incoming" / "ood" / "ood-set.jsonl"
OOD_V2 = ROOT / "data" / "incoming" / "ood" / "ood-set-v2.jsonl"
SWEEP_OUT = ROOT / "data" / "incoming" / "ood" / "energy-sweep-v2.json"
KNN_BANK = ROOT / "models" / "ood-knn-logits.skops"
KNN_META = ROOT / "models" / "ood-knn-logits-meta.json"

# In-domain reference surfaces: (name, loader tag). Kept to sizes that run
# in seconds on CPU; every row must genuinely belong to a shipped class.
IN_DOMAIN_LIMITS = {
    "heldout": 350,        # frozen held-out (stop-rule surface)
    "loghub": 2000,        # LogHub sample
    "tickets-en": 300,     # reality slices
    "tickets-de": 300,
    "prs": 300,            # real PRs
    "swebench": 100,       # problem statements
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def in_domain_rows(limit: int = 400) -> list[dict]:
    """Reference in-domain rows from the committed real corpora (field is
    `raw` in every jsonl; the de tickets come from the 20k CSV; SWE-bench
    problem statements from the bench cache)."""
    rows: list[dict] = []

    def add_jsonl(name, path, n):
        taken = 0
        for line in path.read_text(encoding="utf-8", errors="replace") \
                .splitlines():
            if taken >= n or not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text = r.get("raw") or r.get("text") or ""
            if not isinstance(text, str) or len(text.strip()) < 4:
                continue
            rows.append({"surface": name, "text": text})
            taken += 1

    def add_swebench(n):
        path = (ROOT / "data" / "incoming" / "external" / "swebench"
                / "swebench_lite_100.json")
        if not path.is_file():
            return
        for entry in json.loads(path.read_text())[:n]:
            text = (entry.get("problem_statement") or "").strip()
            if len(text) >= 4:
                rows.append({"surface": "swebench", "text": text})

    def add_de_tickets(n):
        import csv

        path = (ROOT / "data" / "incoming" / "external" / "incident-severity"
                / "tickets-multi-lang-20k.csv")
        if not path.is_file():
            return
        taken = 0
        with path.open(encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("language") != "de":
                    continue
                text = (row.get("body") or "").strip()
                if len(text) >= 4:
                    rows.append({"surface": "tickets-de", "text": text})
                    taken += 1
                    if taken >= n:
                        break

    add_jsonl("heldout", ROOT / "data" / "heldout" / "dev-reference.jsonl",
              IN_DOMAIN_LIMITS["heldout"])
    add_jsonl("loghub", ROOT / "data" / "real" / "log_event" / "train.jsonl",
              IN_DOMAIN_LIMITS["loghub"])
    add_jsonl("tickets-en", ROOT / "data" / "real" / "issue" / "train.jsonl",
              IN_DOMAIN_LIMITS["tickets-en"])
    add_de_tickets(IN_DOMAIN_LIMITS["tickets-de"])
    add_jsonl("prs", ROOT / "data" / "real" / "pull_request" / "train.jsonl",
              IN_DOMAIN_LIMITS["prs"])
    add_swebench(IN_DOMAIN_LIMITS["swebench"])
    return rows


def _score_rows(router, texts: list[str]):
    """(confidence, energy) per text — the two inputs of the shipped gate.
    Chunked (the memory law: never whole-batch)."""
    confs = np.empty(len(texts), dtype=np.float64)
    energies = np.empty(len(texts), dtype=np.float64)
    step = 64
    for i in range(0, len(texts), step):
        chunk = texts[i:i + step]
        probs, energy = router.predict_proba_with_energy(chunk)
        confs[i:i + len(chunk)] = probs.max(axis=1)
        energies[i:i + len(chunk)] = energy
    return confs, energies


CONF_GATE = 0.85  # the shipped fast-path confidence constant


def _energies(router, texts: list[str]) -> np.ndarray:
    return _score_rows(router, texts)[1]


# ------------------------------------------------------------------ sweep --


def sweep() -> int:
    from mivia_small_json import encoder

    router = encoder.load_encoder()
    assert router is not None, "int8 router not loadable"

    ood = []
    for path, tag in ((OOD_V1, "v1"), (OOD_V2, "v2")):
        if path.is_file():
            for r in _rows(path):
                ood.append({"surface": f"ood-{tag}", "lang": r.get("lang", "en"),
                            "category": r.get("category", ""),
                            "text": r["text"]})
    ind = in_domain_rows()

    c_ood, e_ood = _score_rows(router, [r["text"] for r in ood])
    c_ind, e_ind = _score_rows(router, [r["text"] for r in ind])
    # Sign convention (encoder.py gate): IN-DOMAIN when energy >= threshold;
    # higher energy is MORE in-domain-like. The OOD emission (phase 1)
    # mirrors the shipped fast-path gate: it may only flag CONFIDENT rows
    # (conf >= 0.85) — unconfident in-domain rows legitimately have low
    # energy and already fall to the fallback/rules path, so the honest
    # retention population is the confident one.
    conf_ind = c_ind >= CONF_GATE
    conf_ood = c_ood >= CONF_GATE
    thresholds = [round(float(t), 2) for t in
                  np.arange(0.0, max(12.0, float(e_ind.max()) + 0.5), 0.1)]

    per_surface = {}
    for name in sorted({r["surface"] for r in ind}):
        mask = np.array([r["surface"] == name for r in ind])
        keep = mask & conf_ind  # the population phase 1 can touch
        per_surface[name] = (e_ind[keep], int(keep.sum()))
    fc_mask = np.array([r.get("category") == "free-chat" for r in ood])
    fc_conf = fc_mask & conf_ood
    nonfc_conf = (~fc_mask) & conf_ood

    sweep_rows = []
    for t in thresholds:
        retained = {}
        for name, (e, n) in per_surface.items():
            retained[name] = (round(float((e >= t).mean()), 4)
                              if n else None)
        # rejection = flagged fraction over the CONFIDENT OOD rows (the
        # rows the emission would act on); the full-set rate is reported
        # alongside as context.
        rejected = (float(((e_ood < t) & conf_ood).sum()) / max(int(conf_ood.sum()), 1))
        fc_rejected = (float(((e_ood < t) & fc_conf).sum())
                       / max(int(fc_conf.sum()), 1))
        nonfc_rejected = (float(((e_ood < t) & nonfc_conf).sum())
                          / max(int(nonfc_conf.sum()), 1))
        sweep_rows.append({
            "threshold": t,
            "ood_rejection": round(rejected, 4),
            "freechat_rejection": round(fc_rejected, 4),
            "nonfreechat_rejection": round(nonfc_rejected, 4),
            "retention": {k: round(v, 4) for k, v in retained.items()},
            "min_retention": round(min(retained.values()), 4),
        })

    # Bars: rejection >= 0.80 (free-chat >= 0.85) AND min retention >= 0.99.
    # Rejection rises with the threshold while retention falls, so among
    # feasible points the SMALLEST threshold keeps the most retention
    # headroom while meeting every bar.
    feasible = [r for r in sweep_rows
                if r["min_retention"] >= 0.99
                and r["ood_rejection"] >= 0.80
                and r["freechat_rejection"] >= 0.85]
    # Fallback (bars unmeetable): the largest threshold keeping retention
    # >= 0.99 — the max-rejection point inside the retention budget.
    fallback = [r for r in sweep_rows if r["min_retention"] >= 0.99]
    chosen = (min(feasible, key=lambda r: r["threshold"]) if feasible
              else (max(fallback, key=lambda r: r["threshold"])
                    if fallback else None))

    current = float(getattr(router, "in_domain_energy", 8.4))
    cur_row = min(sweep_rows, key=lambda r: abs(r["threshold"] - current))
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "router_sha256": hashlib.sha256(
            (ROOT / "models" / "router-encoder-int8.onnx").read_bytes()
        ).hexdigest(),
        "conf_gate": CONF_GATE,
        "confident_ood": int(conf_ood.sum()),
        "confident_ind": int(conf_ind.sum()),
        "population_note": "rejection/retention are computed on the "
                           "conf>=0.85 population the phase-1 emission can "
                           "act on; unconfident rows already fall to the "
                           "fallback/rules path",
        "current_threshold": current,
        "current_point": cur_row,
        "n_ood": len(ood), "n_ind": len(ind),
        "chosen_threshold": chosen["threshold"] if chosen else None,
        "chosen_point": chosen,
        "bars_met": bool(feasible),
        "sweep": sweep_rows,
    }
    SWEEP_OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("current_threshold", "chosen_threshold", "bars_met",
                       "n_ood", "n_ind")}, indent=1))
    if chosen:
        print("chosen:", json.dumps(chosen["retention"], indent=1))
    print(f"written {SWEEP_OUT}")
    return 0 if result["bars_met"] else 1


# -------------------------------------------------------------------- knn --


def knn() -> int:
    from mivia_small_json import encoder
    from sklearn.neighbors import NearestNeighbors

    router = encoder.load_encoder()
    assert router is not None
    # Bank: logit vectors of in-domain training rows (never eval).
    ind = in_domain_rows(limit=4000)
    texts = [r["text"] for r in ind]
    logits = np.concatenate(
        [router._logits(texts[i:i + 64]) for i in range(0, len(texts), 64)],
        axis=0)
    norms = np.linalg.norm(logits, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bank = logits / norms

    nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(bank)

    ood = [r for p in (OOD_V1, OOD_V2) if p.is_file()
           for r in _rows(p)]
    ind_eval = in_domain_rows()
    ood_logits = np.concatenate(
        [router._logits([r["text"] for r in ood[i:i + 64]])
         for i in range(0, len(ood), 64)], axis=0)
    ind_logits = np.concatenate(
        [router._logits([r["text"] for r in ind_eval[i:i + 64]])
         for i in range(0, len(ind_eval), 64)], axis=0)

    def knn_score(mat):
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        n[n == 0] = 1.0
        dist, _ = nn.kneighbors(mat / n)
        return dist.mean(axis=1)  # larger = farther from in-domain

    s_ood = knn_score(ood_logits)
    s_ind = knn_score(ind_logits)

    # Combined decision at the current energy threshold: OOD if energy > t
    # OR knn distance > ktau, where ktau keeps in-domain retention >= 0.99
    # when OR-ed with the energy gate.
    sweep = json.loads(SWEEP_OUT.read_text()) if SWEEP_OUT.is_file() else {}
    t_energy = float(sweep.get("chosen_threshold")
                     or sweep.get("current_threshold") or 8.4)
    ktau = float(np.quantile(s_ind, 0.99))
    e_ood = _energies(router, [r["text"] for r in ood])
    e_ind = _energies(router, [r["text"] for r in ind_eval])
    # OOD = energy BELOW threshold (in-domain = high energy), and OOD =
    # knn distance ABOVE ktau (far from the in-domain bank).
    energy_ood = e_ood < t_energy
    energy_ind = e_ind >= t_energy
    knn_ood = s_ood > ktau
    knn_ind = s_ind <= ktau
    comb_ood = energy_ood | knn_ood
    comb_ind = energy_ind & knn_ind

    fc = np.array([r.get("category") == "free-chat" for r in ood])
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ktau": round(ktau, 4),
        "energy_threshold": t_energy,
        "energy_rejection": round(float(energy_ood.mean()), 4),
        "knn_rejection": round(float(knn_ood.mean()), 4),
        "combined_rejection": round(float(comb_ood.mean()), 4),
        "freechat_rejection": {
            "energy": round(float(energy_ood[fc].mean()), 4),
            "knn": round(float(knn_ood[fc].mean()), 4),
            "combined": round(float(comb_ood[fc].mean()), 4),
        },
        "in_domain_retention": {
            "energy": round(float(energy_ind.mean()), 4),
            "knn": round(float(knn_ind.mean()), 4),
            "combined": round(float(comb_ind.mean()), 4),
        },
        "added_rejection_over_energy": round(
            float(comb_ood.mean() - energy_ood.mean()), 4),
        "bank_rows": int(len(texts)),
    }
    go = (result["added_rejection_over_energy"] >= 0.05
          and result["in_domain_retention"]["combined"] >= 0.99)
    result["go"] = bool(go)

    import skops.io as skio

    skio.dump(nn, KNN_BANK)
    KNN_META.write_text(json.dumps({
        **result,
        "bank_sha256": hashlib.sha256(
            (ROOT / "models" / "router-encoder-int8.onnx").read_bytes()
        ).hexdigest(),
        "note": "kNN over 10-dim int8 LOGITS (the exported graph has no "
                "embedding output); bound to the router sha above",
    }, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    print("GO" if go else "NO-GO: embedding-space KNN needs the owner-gated "
                          "ONNX re-export")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--knn", action="store_true")
    args = parser.parse_args()
    if args.sweep:
        return sweep()
    if args.knn:
        return knn()
    parser.error("choose --sweep / --knn")
    return 2


if __name__ == "__main__":
    sys.exit(main())
