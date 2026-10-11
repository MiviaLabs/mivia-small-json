"""QW-C: Platt-calibrate select() confidence + conformal abstain threshold.

Four phases (run in order; the reserve must be committed before --eval):

    --build-reserve  construct the fit/reserve row sets, commit the frozen
                     reserve (data/heldout/select-calibration/) FIRST
    --fit            fit Platt + tau on the fit half, persist artifacts
    --eval           ONE-SHOT evaluation on the frozen reserve
    --fit-many       refit the conformal tau for the MANY-CANDIDATE band
                     (2026-10-11; see below)

Row construction (documented, deterministic):
- BFCL dev pairs (never trained on): each positive row becomes a selection
  row (gold tool + 3 distractor tools sampled from the dev negative pool).
- Prose rows: 8 domains x templated symptom texts, gold = the domain whose
  keyword signal the text carries; candidates = gold + 3 other domains.
  Fit/reserve assignment by sha256(row)%100 < 60 in BOTH sources.

The rank artifact bytes are never touched; the calibrator is a monotone
post-hoc map (research: Platt for rank scores, not temperature) persisted
as models/select-calibration.skops bound to the rank artifact sha.

--fit-many (the abstain-all-at-large-K fix): the shipped tau 0.7030 was
fitted on the K=4 selection population, and a softmax top-1 over 77
candidates tops out far below it — measured 600/600 abstain on Banking77
test, 0% called from K=8 up (.scratch/zs-diag/kband_measure.py). This
phase re-derives the conformal tau per measured candidate count (the
shape key is the candidate-count band), on TRAIN-split selections where
the gold answer is known:

    K=8/16/32   Banking77 train.csv, seed-42 n=600/point, gold + K-1
                seeded distractors, bench-shape descriptions
    K=77        Banking77 train.csv, seed-42 n=600, all 77 intents
    K=151       CLINC150 train.jsonl, seed-42 n=600, all 151 intents

Each point runs through the SHARED shipped Platt map (monotone; the skops
bytes are untouched) with the same called-risk rule at alpha=0.12 as the
small-band fit, and lands in models/select-calibration-meta.json under
bands.many.tau_by_k (sidecar refreshed). select() serves the point whose
K is nearest the served candidate count by |log ratio|; K <= 4 keeps the
shipped small-band tau.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))  # noqa: E402

PAIRS = ROOT / "data" / "incoming" / "external" / "bfcl" / "pairs.jsonl"
RESERVE_DIR = ROOT / "data" / "heldout" / "select-calibration"
RESERVE = RESERVE_DIR / "reserve.jsonl"
RESERVE_META = RESERVE_DIR / "reserve.json"
CAL_SKOPS = ROOT / "models" / "select-calibration.skops"
CAL_META = ROOT / "models" / "select-calibration-meta.json"
B77_DIR = ROOT / "data" / "incoming" / "external" / "banking77"
CLINC_DIR = ROOT / "data" / "incoming" / "external" / "clinc150"

# ---------------------------------------------------------------------------
# Head registry (OWNER WIRING DECREE 2026-10-11, --head): the calibration
# for the wired zero-shot select-rank head is fitted with --head select-rank
# into its OWN meta keyed by head (models/select-rank-calibration-*) — the
# legacy meta stays byte-identical for the guard-rank surface/rollback.
# Every phase reads its head's files EXPLICITLY (never the module-level
# loader, which follows the ACTIVE head on the wired tree): the old numbers
# stay reproducible byte-for-byte with --head legacy.
# ---------------------------------------------------------------------------
HEADS = {
    "legacy": {
        "skops": CAL_SKOPS,
        "meta": CAL_META,
        "rank_artifact": "models/tool-match-int8.onnx",
        "rank_sha_of": "DEFAULT_MODEL",
        "small_alpha": 0.10,  # the shipped small-band fit's alpha
        "note": "guard rank artifact (pre-decree select surface)",
    },
    "select-rank": {
        "skops": ROOT / "models" / "select-rank-calibration.skops",
        "meta": ROOT / "models" / "select-rank-calibration-meta.json",
        "rank_artifact": "models/select-rank-int8.onnx",
        "rank_sha_of": "SELECT_RANK_MODEL",
        "small_alpha": 0.12,  # the currently registered alpha (QW-C run 2)
        "note": "wired zero-shot select-rank head (r2 candidate 2db0d0ae)",
    },
}

# Many-band refit points. Each: n_fit rows, seed 42, bench-shape descs.
MANY_POINTS = (
    {"k": 8, "dataset": "banking77-train"},
    {"k": 16, "dataset": "banking77-train"},
    {"k": 32, "dataset": "banking77-train"},
    {"k": 77, "dataset": "banking77-train-all"},
    {"k": 151, "dataset": "clinc150-train-all"},
)
MANY_N = 600
MANY_SEED = 42

DOMAINS = {
    "Payments & Billing": ["checkout", "charge", "3d secure", "invoice",
                           "refund", "billing"],
    "Authentication": ["login", "password", "sso", "oauth", "token expired",
                       "sign in"],
    "Infrastructure": ["kubernetes", "cluster", "deploy", "server", "cpu",
                       "disk full", "instance"],
    "Frontend": ["css", "button", "layout", "render", "browser", "page load"],
    "Database": ["query", "migration", "index", "postgres", "replica",
                 "deadlock"],
    "Security": ["credential", "vulnerability", "cve", "exploit", "leak",
                 "unauthorized access"],
    "Mobile": ["ios", "android", "app crash", "push notification",
               "app store", "battery"],
    "Documentation": ["readme", "docs page", "typo in the guide",
                      "missing example", "changelog", "tutorial"],
}


def _row_hash(row: dict) -> int:
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return int(hashlib.sha256(blob.encode()).hexdigest(), 16) % 100


def build_rows() -> tuple[list[dict], list[dict]]:
    """(fit_rows, reserve_rows) — BFCL-dev selection rows + prose rows."""
    dev = [json.loads(line) for line in PAIRS.read_text().splitlines()
           if line.strip() and json.loads(line)["split"] == "dev"]
    positives = [r for r in dev if r["label"] == 1]
    neg_tools = sorted({r["tool"] for r in dev if r["label"] == 0})
    rng = random.Random(20261009)

    rows: list[dict] = []
    for r in positives:
        distractors = rng.sample(neg_tools, 3)
        cands = [r["tool"]] + distractors
        rng.shuffle(cands)
        rows.append({"source": "bfcl_dev", "text": r["request"],
                     "candidates": cands, "gold": r["tool"]})

    for i, (domain, keywords) in enumerate(DOMAINS.items()):
        others = [d for d in DOMAINS if d != domain]
        for j in range(150):  # 1,200 prose rows total
            kws = [keywords[(j * 3 + off) % len(keywords)] for off in range(3)]
            text = (f"Customer report {i * 150 + j:04d}: the {kws[0]} flow "
                    f"breaks when the {kws[1]} step runs after a "
                    f"{kws[2]} change; users are blocked since this week.")
            cands = [domain] + rng.sample(others, 3)
            rng.shuffle(cands)
            rows.append({"source": "prose_constructed", "text": text,
                         "candidates": cands, "gold": domain})

    fit = [r for r in rows if _row_hash(r) < 60]
    reserve = [r for r in rows if _row_hash(r) >= 60]
    return fit, reserve


def top1_raw(matcher, row: dict) -> tuple[str, float]:
    # Pre-Platt softmax. select() calibrates by default; fitting the map
    # on that output would train it on itself.
    choice, conf = matcher.select(
        row["text"], row["candidates"], calibrated=False)
    return choice, float(conf)


# ------------------------------------------------------------- head plumbing


def head_matcher(name: str):
    """The phase's rank matcher, EXPLICIT-PATH.

    Never the module-level surface resolvers: on the wired tree
    ``load_matcher().select`` serves the wired select-rank head, so the
    legacy phases must score the guard rank bytes directly (the same
    plain-candidate path ``SplitToolMatcher.rank.select`` ran pre-decree)
    and the select-rank phases must score the promoted head bytes.
    """
    from mivia_small_json import tool_match as tm

    if name == "select-rank":
        m = tm.load_single_matcher(
            tm.SELECT_RANK_MODEL, tokenizer_path=tm.SELECT_RANK_TOKENIZER)
        if m is None:
            raise SystemExit("select-rank head not loadable — promote it "
                             "first (models/select-rank-int8.onnx)")
        return m
    m = tm.load_single_matcher(tm.DEFAULT_MODEL,
                               tokenizer_path=tm.DEFAULT_TOKENIZER)
    if m is None:
        raise SystemExit("guard rank artifact not loadable")
    return m


def head_cal(name: str):
    """Explicit (lr, tau_small, tau_by_k) for one head, fixed paths.

    Mirrors the module loader's verification/staleness contract without
    following the ACTIVE head: ``--head legacy`` must reproduce the old
    numbers even on the wired tree (where load_select_calibrator now
    serves the select-rank map)."""
    from mivia_small_json.extract_learned import _load_skops_verified
    from mivia_small_json import tool_match as tm

    spec = HEADS[name]
    meta_path = spec["meta"]
    if not tm._meta_verified(meta_path):
        raise SystemExit(f"{meta_path.name} failed its sha256 sidecar "
                         "check")
    meta = json.loads(meta_path.read_text())
    model = getattr(tm, spec["rank_sha_of"])
    if meta.get("rank_sha256") != tm.integrity.sha256_file(model):
        raise SystemExit(f"{spec['rank_artifact']} changed since the fit — "
                         "re-run --fit first")
    lr = _load_skops_verified(spec["skops"], f"{name} select calibrator")
    if lr is None:
        raise SystemExit("calibrator skops unavailable — run --fit first")
    tau_by_k = None
    points = ((meta.get("bands") or {}).get("many") or {}).get("tau_by_k")
    if isinstance(points, list) and points:
        tau_by_k = tuple(sorted((int(p["k"]), float(p["tau"]))
                                for p in points))
    return lr, float(meta["tau"]), tau_by_k


def _judged(lr, raw: float) -> float:
    from mivia_small_json import tool_match as tm

    return tm._apply_select_calibration(raw, (lr, 1.0, None))


# ---------------------------------------------------------------- phases --


def build_reserve() -> int:
    _, reserve = build_rows()
    RESERVE_DIR.mkdir(parents=True, exist_ok=True)
    with RESERVE.open("w") as fh:
        for row in reserve:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    sha = hashlib.sha256(RESERVE.read_bytes()).hexdigest()
    RESERVE_META.write_text(json.dumps({
        "purpose": "QW-C select-calibration one-shot reserve (frozen)",
        "reserved_utc": datetime.now(timezone.utc).isoformat(),
        "n": len(reserve),
        "sources": sorted({r["source"] for r in reserve}),
        "construction": "BFCL dev positives + seeded prose templates; "
                        "sha256(row)%100>=60; disjoint from the fit half "
                        "and from every training set (dev was never "
                        "trained on)",
        "bfcl_pairs_sha256": hashlib.sha256(PAIRS.read_bytes()).hexdigest(),
        "sha256": sha,
    }, indent=1) + "\n")
    print(f"reserve: {len(reserve)} rows -> {RESERVE}")
    print(f"sha256 {sha[:16]}... (commit before --eval)")
    return len(reserve)


def _fit_called_risk_threshold(calibrated, correct, alpha: float) -> float:
    """Smallest tau whose AMONG-CALLED wrong rate meets alpha (with the
    finite-sample correction). NOT fit_risk_threshold: that function bounds
    the wrong rate over ALL rows (the guard's false-fire budget), while the
    QW-C bar is conditional on the called set. First fit used the wrong
    loss (recorded FAIL 2026-10-09); this is the corrected semantics."""
    import numpy as np

    c = np.asarray(calibrated, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    for tau in np.unique(c):
        called = ok[c >= tau]
        if called.size == 0:
            continue
        wrong = 1.0 - float(called.mean())
        n = called.size
        if (n / (n + 1)) * wrong + 1 / (n + 1) <= alpha:
            return float(tau)
    return 1.0


def fit(head: str = "legacy") -> None:
    from mivia_small_json import tool_match
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    spec = HEADS[head]

    # The reserve pins the pairs file it was built from; a regenerated
    # pairs.jsonl could shift a row across the fit/reserve hash split, so
    # a mismatch refuses the fit (review P2: reserve/fit disjointness).
    if RESERVE_META.is_file():
        meta = json.loads(RESERVE_META.read_text())
        want = meta.get("bfcl_pairs_sha256")
        if want and want != hashlib.sha256(PAIRS.read_bytes()).hexdigest():
            raise SystemExit(
                "pairs.jsonl changed since the reserve was frozen — rebuild "
                "the reserve (and re-run the one-shot) before fitting")

    matcher = head_matcher(head)
    fit_rows, _ = build_rows()
    xs, ys = [], []
    for row in fit_rows:
        choice, conf = top1_raw(matcher, row)
        xs.append([conf])
        ys.append(1 if choice == row["gold"] else 0)
    xs_np = np.asarray(xs, dtype=np.float64)
    ys_np = np.asarray(ys, dtype=np.int64)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(xs_np, ys_np)

    calibrated = lr.predict_proba(xs_np)[:, 1]
    tau = _fit_called_risk_threshold(calibrated, ys_np,
                                     alpha=spec["small_alpha"])

    import skops.io as skio

    skio.dump(lr, spec["skops"])
    rank_sha = tool_match.integrity.sha256_file(
        getattr(tool_match, spec["rank_sha_of"]))

    meta = {
        "alpha": spec["small_alpha"],
        "tau": tau,
        "n_fit": int(len(ys_np)),
        "fit_positives": int(ys_np.sum()),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "rank_artifact": spec["rank_artifact"],
        "rank_sha256": rank_sha,
        "ece_before": _ece(xs_np[:, 0], ys_np),
        "ece_after": _ece(calibrated, ys_np),
    }
    if head == "select-rank":
        meta["head"] = "select-rank (zero-shot-select-r2 candidate " \
                       "2db0d0aed91f1681…, OWNER WIRING DECREE 2026-10-11)"
        meta["serves"] = "matcher.select() / POST /v1/select (tool head)"
        meta["companion_meta"] = "models/select-calibration-meta.json " \
                                 "(the guard-rank surface; legacy, " \
                                 "byte-identical, rollback)"
        meta["band_note"] = (
            "Second calibration meta keyed by head (decree option b): the "
            "legacy select-calibration-meta.json describes the guard rank "
            "bytes and is never served on this head; this map was fitted "
            "on the WIRED head's own confidence distribution over the same "
            "BFCL-dev + prose fit half (small band) and Banking77/CLINC150 "
            "train-split selections (many bands).")
    meta_path = spec["meta"]
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    meta = json.loads(meta_path.read_text())
    print(f"[{head}] fit: n={len(ys_np)} pos={ys_np.sum()} tau={tau:.4f} "
          f"ECE {meta['ece_before']:.4f} -> {meta['ece_after']:.4f} "
          f"alpha={spec['small_alpha']}")
    print(f"wrote {spec['skops'].name} + {meta_path.name} "
          f"(bound to {spec['rank_artifact']} sha)")


def _ece(conf, correct, bins: int = 10) -> float:
    import numpy as np

    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = conf.size
    ece = 0.0
    for i in range(bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1]) \
            if i else (conf >= 0.0) & (conf <= edges[1])
        if not mask.any():
            continue
        ece += mask.sum() / total * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def evaluate(head: str = "legacy") -> int:
    """ONE-SHOT on the frozen reserve. No threshold may be tuned here."""

    meta = json.loads(RESERVE_META.read_text())
    actual_sha = hashlib.sha256(RESERVE.read_bytes()).hexdigest()
    assert actual_sha == meta["sha256"], "reserve drift — eval refused"
    rows = [json.loads(line) for line in RESERVE.read_text().splitlines()
            if line.strip()]

    matcher = head_matcher(head)
    lr, tau, _tau_by_k = head_cal(head)
    if lr is None:
        raise SystemExit("select calibrator unavailable — run --fit first")

    raws, ys, calibrated, picks = [], [], [], []
    for row in rows:
        choice, conf = top1_raw(matcher, row)
        correct = choice == row["gold"]
        raws.append(conf)
        ys.append(1 if correct else 0)
        calibrated.append(_judged(lr, conf))
        picks.append((choice, row["gold"], conf))

    import numpy as np

    raws_np, cal_np = np.asarray(raws), np.asarray(calibrated)
    ys_np = np.asarray(ys, dtype=np.int64)
    # wrong-pick rate among non-abstained picks (the shipped rule:
    # status "abstain" when calibrated < tau) — reproduced exactly here.
    called = [(ch == gold) for (ch, gold, _), c in zip(picks, calibrated)
              if c >= tau]
    ece_before = _ece(raws_np, ys_np)
    ece_after = _ece(cal_np, ys_np)

    result = {
        "head": head,
        "reserve_n": len(rows),
        "reserve_sha256": actual_sha,
        "alpha": 0.10,
        "tau": tau,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "ece_reduction": round(1 - ece_after / ece_before, 4) if ece_before else 0.0,
        "wrong_pick_rate_called": round(1 - float(np.mean(called)), 4) if called else 1.0,
        "coverage_called": round(len(called) / len(rows), 4),
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(result, indent=1))

    bars = {
        "ece_reduction>=0.5": result["ece_reduction"] >= 0.5,
        "wrong_pick<=0.11": result["wrong_pick_rate_called"] <= 0.11,
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"
    print(f"VERDICT {verdict}: {bars}")
    out = ROOT / "reports" / (
        "select-calibration-oneshot.json" if head == "legacy" else
        f"select-calibration-oneshot-{head}.json")
    out.write_text(json.dumps({"verdict": verdict, "bars": bars,
                               **result}, indent=1) + "\n")
    print(f"written {out}")
    return 0 if verdict == "PASS" else 1


def _bench_descs(labels: list[str]) -> dict[str, str]:
    """The bench-shape description both Banking77/CLINC benchmarks use."""
    return {i: f"{i.replace('_', ' ')} ({i.replace('_', ' ')})"
            for i in labels}


def _b77_train_rows() -> list[tuple[str, str]]:
    rows = []
    with (B77_DIR / "train.csv").open(encoding="utf-8", errors="replace",
                                      newline="") as fh:
        import csv

        for r in csv.DictReader(fh):
            t = (r.get("text") or "").strip()
            c = (r.get("category") or r.get("label") or "").strip()
            if t and c:
                rows.append((t, c))
    if len(rows) < MANY_N:
        raise SystemExit(f"banking77 train has {len(rows)} usable rows")
    return rows


def _clinc_train_rows() -> list[tuple[str, str]]:
    rows = []
    with (CLINC_DIR / "train.jsonl").open(encoding="utf-8",
                                          errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            t = (r.get("text") or "").strip()
            lb = (r.get("intent") or "").strip()
            if t and lb:
                rows.append((t, lb))
    if len(rows) < MANY_N:
        raise SystemExit(f"clinc150 train has {len(rows)} usable rows")
    return rows


def fit_many(head: str = "legacy") -> None:
    """Re-derive the conformal tau per candidate-count band (see module
    docstring). Preserves every legacy meta key; refreshes the sidecar."""
    import numpy as np

    from mivia_small_json import tool_match

    spec = HEADS[head]
    lr, tau_small, _ = head_cal(head)

    meta_path = spec["meta"]
    meta = json.loads(meta_path.read_text())
    rank_sha = tool_match.integrity.sha256_file(
        getattr(tool_match, spec["rank_sha_of"]))
    if meta.get("rank_sha256") != rank_sha:
        raise SystemExit(f"{spec['rank_artifact']} changed since the "
                         "small-band fit — re-run --fit (and "
                         "re-adjudicate) first")

    matcher = head_matcher(head)

    b77_rows = _b77_train_rows()
    clinc_rows = _clinc_train_rows()
    # categories.json / labels.json order — the bench's own candidate
    # order, so the K=77/151 points measure the same shape the benches do.
    b77_labels = json.loads((B77_DIR / "categories.json").read_text())
    clinc_labels = json.loads((CLINC_DIR / "labels.json").read_text())
    b77_descs = _bench_descs(b77_labels)
    clinc_descs = _bench_descs(clinc_labels)

    points = []
    curves = {}
    rng = random.Random(MANY_SEED)
    for spec_pt in MANY_POINTS:
        k, dataset = spec_pt["k"], spec_pt["dataset"]
        if dataset == "clinc150-train-all":
            rows = rng.sample(clinc_rows, MANY_N)
            descs = clinc_descs
            cands = None
        else:
            rows = rng.sample(b77_rows, MANY_N)
            descs = b77_descs
            cands = None if dataset == "banking77-train-all" else k
        raws, corr = [], []
        for text, gold in rows:
            if cands is None:
                sel = descs
            else:
                distractors = rng.sample([i for i in descs if i != gold],
                                         cands - 1)
                pool = [gold] + distractors
                rng.shuffle(pool)
                sel = {i: descs[i] for i in pool}
            choice, conf = top1_raw(matcher, {"text": text,
                                              "candidates": sel})
            raws.append(conf)
            corr.append(1 if choice == gold else 0)
        judged = [_judged(lr, c) for c in raws]
        tau = _fit_called_risk_threshold(judged, corr,
                                         alpha=tool_match.SELECT_ABSTAIN_ALPHA)
        called = [(c == 1) for c, j in zip(corr, judged) if j >= tau]
        point = {
            "k": k,
            "tau": round(float(tau), 12),
            "n_fit": len(corr),
            "fit_positives": int(sum(corr)),
            "top1_fit": round(100 * sum(corr) / len(corr), 2),
            "coverage_at_tau": round(len(called) / len(corr), 4),
            "wrong_called_at_tau": round(1 - (sum(called) / len(called)), 4)
            if called else 1.0,
            "alpha": tool_match.SELECT_ABSTAIN_ALPHA,
            "source": dataset,
            "seed": MANY_SEED,
        }
        points.append(point)
        curves[k] = _gating_curve_points(np.asarray(judged),
                                         np.asarray(corr, dtype=np.int64))
        print(json.dumps(point, indent=1), flush=True)

    meta["bands"] = {
        "shape_key": "candidate_count",
        "small": {
            "k_max": tool_match.SELECT_SMALL_BAND_K_MAX,
            "tau": meta["tau"],
            "note": "the small-band fit (BFCL-dev + prose, K=4, gold + 3 "
                    "distractors) on THIS head's own softmax",
        },
        "many": {
            "k_min": tool_match.SELECT_SMALL_BAND_K_MAX + 1,
            "assignment": "serve the fit point with the nearest K by "
                          "|log(k / k_fit)|; ties to the larger K "
                          "(the lower, more conservative support)",
            "platt_map": f"this head's own {spec['skops'].name} "
                         "(monotone, fitted on the K=4 population); each "
                         "tau is conformal on its own point's population "
                         "at the registered alpha",
            "tau_by_k": points,
            "fitted_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    meta["band_note"] = (
        f"{datetime.now(timezone.utc).date().isoformat()} wiring fit: the "
        f"band-aware conformal tau re-derived on the WIRED head's own "
        f"confidence distribution over train-split selections ({head}); "
        "the legacy select-calibration-meta.json is untouched and keeps "
        "serving the guard-rank surface. Confidence gating (status "
        "abstain under the band tau) is the shipped abstention mechanism "
        "for this head — the dedicated nomatch head is NOT wired (its "
        "null signal never fires on near-domain oos; registered "
        "accepted trade).")
    payload = json.dumps(meta, indent=1) + "\n"
    meta_path.write_text(payload)
    (ROOT / "models" / (meta_path.name + ".sha256")).write_text(
        hashlib.sha256(payload.encode()).hexdigest() + "\n")
    print(f"wrote {meta_path.name} + sidecar "
          f"({len(points)} many-band points)")

    # The accuracy/coverage curve comes free from the same scored
    # populations (no second inference pass): one entry per fitted K.
    out = ROOT / "reports" / "select-rank-gating-curve.json"
    payload = {
        "head": head,
        "rank_artifact": spec["rank_artifact"],
        "rank_sha256": rank_sha,
        "platt_map": spec["skops"].name,
        "alpha": tool_match.SELECT_ABSTAIN_ALPHA,
        "population": "TRAIN-split selections, seed 42, n=600/point "
                      "(bench-shape descriptions; never trained on by "
                      "the guard; Banking77/CLINC150 train files)",
        "note": "accuracy/coverage curve of select-when-confident gating "
                "on the wired head; tau_by_k points above are the "
                "operating points shipped in the meta",
        "curves": {str(k): curves[k] for k in sorted(curves)},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out.name}")


def _gating_curve_points(judged, corr, grid: int = 12) -> list[dict]:
    """Coverage/accuracy sweep over a tau grid on one population.

    Deterministic quantile grid over the judged confidence; each row
    reports the called set's size, accuracy and wrong rate. The fitted
    tau's own point (from tau_by_k) is the shipped operating point — this
    curve is the measured frontier around it, never a retune."""
    import numpy as np

    taus = np.unique(np.quantile(judged,
                                 np.linspace(0.0, 1.0, grid + 1)))
    rows = []
    for t in taus:
        called = corr[judged >= t]
        if called.size == 0:
            continue
        rows.append({
            "tau": round(float(t), 6),
            "coverage": round(float(called.size) / float(corr.size), 4),
            "accuracy_called": round(float(called.mean()), 4),
            "wrong_called": round(1 - float(called.mean()), 4),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=sorted(HEADS), default="legacy",
                        help="whose calibration to fit/measure (default "
                        "legacy = the guard rank surface, pre-decree "
                        "semantics; select-rank = the wired zero-shot "
                        "head, OWNER WIRING DECREE 2026-10-11)")
    parser.add_argument("--build-reserve", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--fit-many", action="store_true")
    args = parser.parse_args()
    if args.build_reserve:
        return build_reserve() and 0
    if args.fit:
        fit(args.head)
        return 0
    if args.eval:
        return evaluate(args.head)
    if args.fit_many:
        fit_many(args.head)
        return 0
    parser.error("choose --build-reserve / --fit / --eval / --fit-many")
    return 2


if __name__ == "__main__":
    sys.exit(main())
