# Copyright 2026 MiviaLabs LLC
# SPDX-License-Identifier: Apache-2.0
"""Iteration-4 improvement flywheel — one cycle, end to end.

``run_cycle`` collects fresh records, weak-labels them (label.py tiers),
appends confident rows to ``data/pool/confident.jsonl``, optionally
retrains through the committed train entry, and applies the PR gate
matrix.

Measurement sets have one job each:
- **Development** (``data/synthetic/dev.jsonl`` and the two real dev
  splits) sets trust, thresholds, admission, and every PR gate input.
- **Frozen audit** (``data/heldout``, ``data/wild``) reports final
  performance and supplies exactly one gate: at cycle end a candidate
  that falls below the held-out floors (``gates_pass``) is vetoed —
  a one-shot stop check, never an input to thresholds, label
  admission, or tuning. ``data/wild`` stays reported, never read.

A calibration is tied to the router hash it was measured on. Training
invalidates it, so a retraining cycle measures again after it has decided
(``_refresh_calibration``). Without that step every later cycle would find
no calibration and fall back to the second label tier only.

Known interaction: ``model.main()`` rewrites the artifact before the gate
runs, so a cycle that refuses the artifact still leaves it on disk. The
calibration is then stale and the first tier stays off until a later cycle
is accepted. That fails safe and is reported as ``not-written``, and the
committed-hash check turns red, so it cannot pass unnoticed. Moving the
artifact write behind the gate is the real fix and is out of scope here.

Known constraint: the labeled development sets decide three separate
things — rule trust, the tier 1 calibration, and the PR gate. That is one
set of 800 rows feeding three decisions, so a retrain that improves the
development numbers improves all three at once and the cycle cannot
separate real generalisation from fitting the set it decides on. Held-out
audit data is what distinguishes them: it is measured, it can veto a
candidate at cycle end, and it is never used for anything else.
Narrowing that would need a held-out slice inside development.

PR gate matrix (all must hold for ``decision == "pr"``):
- >= MIN_NEW_CONFIDENT new confident rows this cycle
- development routed-and-valid >= 0.99 (the strict definition: routed to
  the record's true schema AND the emitted object passes the strict
  validator; the weaker schema-valid rate is reported beside it, never
  gated on)
- development macro F1 >= previous + MACRO_GAIN_MIN, skipped as
  "saturated" once the development macro has no headroom left
- no development class drops more than PER_CLASS_DROP_MAX
- held-out routed-and-valid >= HELDOUT_FLOOR and held-out macro F1 >=
  HELDOUT_F1_FLOOR (the one-shot final veto; skipped when held-out was
  not measured)
- PSI(class mix of this cycle's PRE-ADMISSION label pool vs the training
  mix) < PSI_BLOCK; the admitted slice's PSI is a reporting line beside it
- artifact <= SIZE_BUDGET

Cycle state lives in ``data/pool``: ``hard.jsonl`` (the cycle's hard
rows, capped) and ``manifest.json`` (per-cycle counts, hashes, development
F1 baselines and train-mix shares for PSI). The previous-cycle constants
are read from the manifest; the module defaults below are the migration
values used only when no manifest exists yet.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import bootstrap, collect, label, model, registry
from .drift import psi as _psi

# The default cycle root is the caller's working directory: a cycle
# reads and writes inside the tree it runs in. This is NOT the package's
# repository root (model.ROOT) — two different paths that once shared
# one name. flywheel.main() refuses to run where no data/ directory
# exists, so an empty cycle report cannot land in the wrong place.
CYCLE_ROOT = Path(".")
POOL = CYCLE_ROOT / "data" / "pool"
CONFIDENT = POOL / "confident.jsonl"
HARD = POOL / "hard.jsonl"
MANIFEST = POOL / "manifest.json"
SIZE_BUDGET = 1_000_000
HARD_CAP = 300

MIN_NEW_CONFIDENT = 50
MACRO_GAIN_MIN = 0.01
PER_CLASS_DROP_MAX = 0.01
PSI_BLOCK = 0.2
# A candidate that falls below these on the frozen held-out set is vetoed,
# whatever its development numbers say — the frozen audit's one-shot final
# stop check at cycle end, never an input to thresholds, admission, or
# tuning. Set above the old 0.99 floor so a sub-point regression (the
# harvest retrain cost 0.29pp) fails loudly.
HELDOUT_FLOOR = 0.999
HELDOUT_F1_FLOOR = 0.999

# Migration defaults, used when data/pool/manifest.json does not exist
# yet. Measured on the labeled development sets with the committed
# artifact. A shipped cycle writes them into the manifest and updates
# them from its own measured development numbers. Wild numbers are not
# a baseline: frozen audit results must not steer the next cycle.
DEFAULT_BASELINES = {
    "prev_dev_macro": 0.9992,
    "prev_dev_per_class": {"issue": 0.9987, "log_event": 1.0, "commit": 0.998,
                           "pull_request": 1.0},
}
DEFAULT_TRAIN_MIX = {"issue": 0.27, "log_event": 0.25, "commit": 0.27,
                     "pull_request": 0.21}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_or_empty(path: Path) -> dict:
    """Parse a JSON object from ``path``; a missing or bad file reads empty."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def read_pool_state(pool: Path = POOL) -> dict:
    """Baselines + train-mix shares from the pool manifest (or defaults)."""
    state = {
        "baselines": {k: (dict(v) if isinstance(v, dict) else v)
                      for k, v in DEFAULT_BASELINES.items()},
        "train_mix": dict(DEFAULT_TRAIN_MIX),
    }
    manifest = pool / "manifest.json"
    if manifest.is_file():
        stored = _read_json_or_empty(manifest)
        for key in ("baselines", "train_mix"):
            if isinstance(stored.get(key), dict):
                state[key].update(stored[key])
        # A baseline left behind by an older cycle is dropped, not
        # carried into the next decision. Only recognized keys survive.
        state["baselines"] = {k: v for k, v in state["baselines"].items()
                              if k in DEFAULT_BASELINES}
    return state


def _write_pool_state(pool: Path, hard: list[dict], report: dict,
                      state: dict) -> None:
    """Persist the cycle: hard.jsonl (capped) + manifest.json."""
    pool.mkdir(parents=True, exist_ok=True)
    hard_path = pool / "hard.jsonl"
    hard_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hard[:HARD_CAP]),
        encoding="utf-8",
    )
    # A shipped PR resets the baselines to this cycle's measured
    # development F1; a report-only cycle keeps them for the next
    # attempt. Wild numbers never become a baseline.
    if report.get("decision") == "pr":
        state["baselines"]["prev_dev_macro"] = report["dev"]["f1_macro"]
        state["baselines"]["prev_dev_per_class"] = dict(
            report["dev"]["f1_per_class"])
    manifest_path = pool / "manifest.json"
    manifest = _read_json_or_empty(manifest_path)
    cycles = manifest.get("cycles") if isinstance(manifest.get("cycles"), list) \
        else []
    cycles.append({
        "date": report.get("date"),
        "decision": report.get("decision"),
        "new_confident": report.get("new_confident"),
        "pool_total": report.get("pool_total"),
        "hard": len(hard),
        "collect": report.get("collect"),
        "label": report.get("label"),
        "trust": report.get("trust"),
    })
    manifest = {
        "version": 1,
        "baselines": state["baselines"],
        "train_mix": state["train_mix"],
        "cycles": cycles,
        "hashes": {
            "confident.jsonl": _sha256_file(pool / "confident.jsonl"),
            "hard.jsonl": _sha256_file(hard_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")


def _append_confident(rows: list[dict], path: Path | None = None) -> int:
    """Append new confident rows (dedup by id); returns how many were new.

    Creates the parent directory if missing, so a fresh clone without
    the directory still completes a cycle.
    """
    path = Path(path) if path is not None else CONFIDENT
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["id"] for r in _read_jsonl(path)}
    new = [r for r in rows if r["id"] not in existing]
    with path.open("a", encoding="utf-8") as fh:
        for r in new:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def dev_gain_headroom(prev_dev_macro: float) -> bool:
    """True while the development macro gain gate can still be satisfied.

    The development macro is near its ceiling, so a fixed +MACRO_GAIN_MIN
    step can be unreachable. When no headroom is left the gate is
    reported as saturated instead of blocking every cycle forever; the
    per-class regression floor still applies.
    """
    return prev_dev_macro + MACRO_GAIN_MIN <= 1.0


def gates_pass(report: dict, baselines: dict | None = None,
               train_mix: dict | None = None) -> tuple[bool, str]:
    """Apply the PR gate matrix. Development numbers decide; the frozen
    held-out set can veto.

    ``report["heldout"]`` is normally measurement-only, but a strict
    routed-and-valid below ``HELDOUT_FLOOR`` vetoes the candidate: a
    development-only gain that hurts the frozen set must not ship.
    ``report["wild"]`` stays reported, never read.
    """
    if baselines is None or train_mix is None:
        stored = read_pool_state()
        baselines = stored["baselines"] if baselines is None else baselines
        train_mix = stored["train_mix"] if train_mix is None else train_mix
    prev_macro = baselines["prev_dev_macro"]
    prev_per_class = baselines["prev_dev_per_class"]
    dev = report["dev"]
    per_class = dev["f1_per_class"]
    # The drop gate measures the UNION of baseline and candidate classes.
    # Iterating the baseline keys only let a NEW class escape the gate
    # entirely — a class at F1 0.0 in its own development world passed
    # while the saturated macro gate looked elsewhere (V1, 2026-10-10).
    drops = []
    for cls in set(prev_per_class) | set(per_class):
        base = prev_per_class.get(cls)
        if base is None:
            # A class the previous cycle never measured has no drop to
            # compare against; it is held to the near-perfect bar the
            # measured baselines carry instead of escaping measurement.
            base = 1.0
        # A baseline class missing from the candidate's map is a total
        # drop, not an absence to skip.
        current = per_class.get(cls, 0.0)
        if current < base - PER_CLASS_DROP_MAX:
            label_txt = (f"{cls} {base:.4f} -> {current:.4f}" if cls in per_class
                         else f"{cls} {base:.4f} -> missing")
            drops.append(label_txt)
    heldout_strict = (report.get("heldout") or {}).get("routed_and_valid")
    heldout_f1 = (report.get("heldout") or {}).get("f1_macro")
    checks = [
        (report["new_confident"] >= MIN_NEW_CONFIDENT,
         f"new confident {report['new_confident']} < {MIN_NEW_CONFIDENT}"),
        # A report missing the strict metric must FAIL, not silently fall
        # back to the weaker schema-valid rate: the gate reads
        # routed-and-valid or it reads nothing.
        (dev.get("routed_and_valid", 0.0) >= 0.99,
         "development routed-and-valid < 99%"),
        ((not dev_gain_headroom(prev_macro))
         or dev["f1_macro"] >= prev_macro + MACRO_GAIN_MIN,
         f"development macro F1 {dev['f1_macro']:.4f} < prev+gain"),
        (not drops,
         "a development class dropped > 1 pp: " + ", ".join(sorted(drops))),
        (heldout_strict is None or heldout_strict >= HELDOUT_FLOOR,
         f"held-out routed-and-valid {heldout_strict} < {HELDOUT_FLOOR} (veto)"),
        (heldout_f1 is None or heldout_f1 >= HELDOUT_F1_FLOOR,
         f"held-out macro F1 {heldout_f1} < {HELDOUT_F1_FLOOR} (veto)"),
        (report["psi"] < PSI_BLOCK, f"PSI {report['psi']:.3f} >= {PSI_BLOCK}"),
        (report["artifact_bytes"] <= SIZE_BUDGET, "artifact over size budget"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    return (not failed), "; ".join(failed)


def _active_train_mix(train_mix: dict, active_classes) -> dict:
    """Renormalize the committed train mix over the schemas active at
    cycle time, so PSI compares like-for-like worlds.

    The committed manifest mix spans the full 10-class training world,
    while a cycle labels in the (5-)schema world it booted. Comparing
    the raw dicts unions the key sets, so classes absent on the cycle
    side each contribute a full log-ratio term and PSI sits at ~4.46
    forever — the ``pr`` decision was arithmetically unreachable.
    Renormalizing the mix over the active classes keeps the comparison
    honest without touching the manifest file.
    """
    active = set(active_classes)
    sub = {k: v for k, v in train_mix.items() if k in active}
    total = sum(sub.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in sub.items()}


def _eval_set(pairs: list[tuple[dict, str]], router) -> dict:
    """Evaluate (row, gold) pairs — each gold travels beside its row,
    never through object identity.

    A ``None`` router raises: ``classify(..., None)`` would silently fall
    back to the rules layer, and a rules-fallback score recorded as the
    candidate is a wrong-backend measurement wearing the candidate's name.

    Two validity readings: ``valid`` (weak) = the emitted object passes
    the strict validator for the schema it was routed to (template
    escapes do not count); ``routed_and_valid`` (strict) = that AND the
    route matched the record's true schema. Gates use the strict one.
    """
    from sklearn.metrics import f1_score
    from .pipeline import classify

    if router is None:
        raise ValueError(
            "_eval_set needs a loaded router: a None router would score the "
            "rules fallback and record it as the candidate measurement")
    y_true, y_pred, valid, strict = [], [], 0, 0
    for row, gold in pairs:
        obj, meta = classify(row["raw"], router)
        y_true.append(gold)
        y_pred.append(obj.get("type"))
        valid += bool(meta["valid"])
        strict += bool(meta["valid"]) and obj.get("type") == gold
    n = len(pairs)
    if n == 0:
        return {"n": 0, "valid": 0.0, "routed_and_valid": 0.0,
                "f1_macro": 0.0, "f1_per_class": {}}
    classes = sorted(set(y_true) | set(y_pred))
    per_class = {c: round(f1_score(y_true, y_pred, labels=[c], average="macro"), 4)
                 for c in classes}
    macro = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
    return {"n": n, "valid": round(valid / n, 4),
            "routed_and_valid": round(strict / n, 4), "f1_macro": macro,
            "f1_per_class": per_class}


def _dev_pairs(measure: dict) -> list[tuple[dict, str]]:
    """(row, gold) pairs from the trust measurement already in hand.

    The development corpora are read once per cycle. Re-reading them
    here would re-hash every development file a second time.
    """
    return [({"raw": text}, gold) for text, gold in measure["pairs"]]


def _candidate_router(live: Path) -> tuple[object, str | None]:
    """Reload the just-trained candidate artifact, loudly.

    ``load_router`` never raises — a failed load returns None and
    ``classify`` would then silently fall back to the rules layer. A
    rules-fallback score recorded as the candidate would put a
    wrong-backend number in the report under the candidate's name, so a
    failed load is surfaced as an explicit failure string and the caller
    takes NO candidate measurement.
    """
    from .pipeline import load_router

    router = load_router(live)
    if router is None:
        return None, (
            f"candidate router failed to load from {live} (sha-verified "
            "load returned None); no candidate measurement was taken — "
            "the rules fallback was not scored as the candidate")
    return router, None


def _pool_psi(pool_per_class: dict[str, int], mix: dict[str, float]) -> float:
    """PSI of the cycle's PRE-ADMISSION label pool vs the training mix.

    ``pool_per_class`` counts every record the weak labeler marked
    confident this cycle, BEFORE the per-class and share caps thin it.
    The caps are admission policy, not evidence about the world: a class
    flood the caps reject is exactly the drift the gate must see. The
    admitted slice's PSI is a reporting line beside this, never the gate.
    """
    total = sum(pool_per_class.values())
    shares = {cls: count / max(1, total)
              for cls, count in pool_per_class.items()}
    return round(_psi(shares, mix), 4)


def _environment() -> dict:  # noqa: D401
    """The stack a calibration was measured on.

    A probability number means nothing without the versions that
    produced it. Keep the block with the record.
    """
    import platform

    import numpy
    import scipy
    import sklearn
    import skops

    return {"python": platform.python_version(),
            "platform": platform.platform(),
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "skops": skops.__version__, "numpy": numpy.__version__}


def _refresh_calibration(root: Path, router, report: dict) -> dict | None:
    """Re-measure the calibration for the artifact just written.

    Training changes the router hash. That hash is what ties a stored
    calibration to its artifact, so the record is stale the moment the
    artifact changes. Measuring again here keeps the next cycle able to
    admit.

    A measurement that fails is recorded, not raised. A cycle which
    cannot measure is a reportable state, not a crash. Tier 1 then stays
    off next cycle, which is the safe direction.
    """
    try:
        record = label.measure_dev_calibration(root, router)
        # Bind the record to the artifact that was actually measured, by
        # hashing it. Reading the sidecar would only trust a text file.
        # The artifact is the one this cycle wrote, in this root — not the
        # repository's own copy, which is a different file whenever a
        # cycle runs against a scratch directory.
        artifact = model.artifact_path(root)
        if not artifact.is_file():
            raise ValueError(f"no router artifact at {artifact}")
        record["router_sha256"] = hashlib.sha256(
            artifact.read_bytes()).hexdigest()
        record["environment"] = _environment()
    except Exception as exc:  # best effort, same as label_record
        # A cycle must not abort because a measurement could not be taken.
        # The artifact is already rewritten by this point.
        report["calibration_refreshed"] = {"status": "not-measured",
                                           "reason": str(exc)}
        return None
    target = root / label.CALIBRATION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report["calibration_refreshed"] = {**record, "status": "measured"}
    return record


def _retrain_extractors(root: Path) -> dict:
    import subprocess
    import sys

    # Snapshot the three learned-extractor artifacts (plus sidecars) so a
    # crashed or timed-out training run cannot leave a half-written
    # artifact behind: the snapshot is restored before the exception
    # propagates.
    artifact_names = ("level_extractor.skops", "component_extractor.skops",
                      "implicit_severity.skops")
    paths = []
    for name in artifact_names:
        for suffix in ("", ".sha256"):
            paths.append(root / "models" / (name + suffix))
    snapshot = {p: (p.read_bytes() if p.is_file() else None) for p in paths}

    def _restore():
        for p, data in snapshot.items():
            if data is None:
                if p.exists():
                    p.unlink()
            else:
                p.write_bytes(data)

    results = {}
    try:
        for script_name in ("train_level_extractor.py",
                            "train_component_extractor.py",
                            "train_implicit_severity.py"):
            script_path = root / "scripts" / script_name
            if script_path.is_file():
                try:
                    res = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=root,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=1800,
                    )
                except subprocess.TimeoutExpired:
                    _restore()
                    results[script_name] = {
                        "ok": False,
                        "summary": ("timed out after 1800s; the previous "
                                    "extractor artifact was restored"),
                    }
                    continue
                ok = res.returncode == 0
                summary = (
                    res.stdout.strip().splitlines()[-1]
                    if res.stdout.strip()
                    else res.stderr.strip().splitlines()[-1]
                    if res.stderr.strip()
                    else "done"
                )
                if not ok:
                    # A failed trainer may have left a partial artifact on
                    # disk where extract_learned loads it next. The shipped
                    # bytes must survive the failed retrain byte-identically
                    # (V10, 2026-10-10: the plain returncode != 0 branch used
                    # to keep the partial artifact in place).
                    _restore()
                    summary += "; the previous extractor artifact was restored"
                results[script_name] = {"ok": ok, "summary": summary}
    except BaseException:
        _restore()
        raise
    return results


def _snapshot_artifact(live: Path) -> dict:
    """Read the live artifact, sidecar, and lineage record into memory."""
    sidecar = live.parent / (live.stem + ".sha256")
    meta = live.parent / (live.stem + ".trainmeta.json")
    return {
        "artifact": live.read_bytes() if live.is_file() else None,
        "sidecar": sidecar.read_bytes() if sidecar.is_file() else None,
        "meta": meta.read_bytes() if meta.is_file() else None,
    }


def _restore_artifact(live: Path, snapshot: dict) -> None:
    """Put the live artifact back exactly as snapshotted."""
    sidecar = live.parent / (live.stem + ".sha256")
    meta = live.parent / (live.stem + ".trainmeta.json")
    for path, data in ((live, snapshot["artifact"]),
                       (sidecar, snapshot["sidecar"]),
                       (meta, snapshot["meta"])):
        if data is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def run_cycle(
    root: Path = CYCLE_ROOT,
    do_train: bool = False,
    retrain_extractors: bool = False,
) -> dict:
    bootstrap.register_example_schemas()
    pool = root / "data" / "pool"
    state = read_pool_state(pool)
    records, cstats = collect.collect_all(root)
    from .pipeline import load_router
    # The cycle scores the artifact in ITS OWN tree: the caller's root
    # decides which router labels records, which one retrains, and which
    # one the post-train eval reloads. Never this repository's copy.
    router = load_router(model.artifact_path(root))
    # Trust is measured on labeled development data before a single
    # record is labeled. Frozen audit data is never consulted here.
    measure = label.measure_dev_rules(root)
    trust = label._trust(measure)
    # Tier 1 needs a calibration measured on the artifact in the tree.
    # A retrain changes the router hash, so a stale record switches the
    # tier off rather than carrying an old number forward.
    calibration = label.calibration_from_disk(root)
    confident, hard, lstats = label.build_pools(records, router, trust=trust,
                                                calibration=calibration,
                                                measure=measure)
    new_confident = _append_confident(confident, pool / "confident.jsonl")

    report = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "collect": cstats, "label": lstats,
        "trust": lstats["trust"],
        "calibration": (None if calibration is None else {
            "source": calibration.source,
            "n": calibration.n,
            "max_significant_error": calibration.max_abs_error,
            "max_abs_error": calibration.worst_band_error,
            "note": "max_significant_error is the largest band deviation "
                    "that gated this cycle; max_abs_error is the worst "
                    "band of the record that gated it. Both travel in this "
                    "report because an accepted cycle replaces "
                    "data/calibration.json before anyone reads it"}),
        "tier1_enabled": lstats["tier1_enabled"],
        "new_confident": new_confident,
        "pool_total": len(_read_jsonl(pool / "confident.jsonl")),
        # A deterministic sample of this cycle's auto-labeled rows, for a
        # human spot-check of weak-label precision.
        "audit": {"rate": label.AUDIT_RATE,
                  "sample_ids": label.audit_sample(confident)},
    }

    if do_train:
        if retrain_extractors:
            report["extractors"] = _retrain_extractors(root)
        live = model.artifact_path(root)
        # Snapshot the live artifact first. A refused candidate is rolled
        # back to it, so a refused cycle is byte-identical on disk — and
        # so is a cycle that dies mid-train: any exception restores the
        # snapshot before propagating, leaving no ungated artifact.
        snapshot = _snapshot_artifact(live)
        try:
            model.main(root=root)  # training mix picks up data/pool/confident.jsonl
            # A failed candidate load must not become a measurement: the
            # explicit failure entry below is the record of record, and
            # the previous artifact is restored like any refused candidate.
            router, load_failure = _candidate_router(live)
            if load_failure is not None:
                _restore_artifact(live, snapshot)
                report["candidate"] = {"ok": False, "reason": load_failure}
                report["gate_reason"] = load_failure
                report["decision"] = "report-only"
                report["calibration_refreshed"] = {
                    "status": "not-written",
                    "reason": "the candidate artifact failed to load; the "
                              "previous artifact was restored and no "
                              "calibration was written"}
            else:
                report["artifact_bytes"] = live.stat().st_size
                report["dev"] = _eval_set(_dev_pairs(measure), router)
                report["dev_gain_gate"] = (
                    "applicable"
                    if dev_gain_headroom(state["baselines"]["prev_dev_macro"])
                    else "saturated")
                # Frozen audit measurement only. Reported, never gated on.
                heldout = [(r, r["schema"]) for r in _read_jsonl(
                    root / "data" / "heldout" / "test.jsonl")]
                wild = [(r, "issue")
                        for r in _read_jsonl(root / "data" / "wild" / "issue.jsonl")]
                wild += [(r, "commit") for r in _read_jsonl(
                    root / "data" / "wild" / "commit.jsonl")]
                wild += [(r, "log_event") for r in _read_jsonl(
                    root / "data" / "wild" / "platform-logs.jsonl")]
                report["heldout"] = _eval_set(heldout, router)
                report["wild"] = _eval_set(wild, router)
                report["audit_only"] = ["heldout", "wild"]
                # Like-for-like PSI: renormalize the committed mix over the
                # schemas active in this cycle's labeling world (see
                # _active_train_mix); the manifest file stays untouched.
                # The gate reads the PRE-ADMISSION pool (V6): the caps are
                # admission policy, so a shift the caps reject must still
                # block. The admitted slice's PSI is a reporting line.
                mix = _active_train_mix(state["train_mix"], registry.active())
                report["psi"] = _pool_psi(lstats.get("pool_per_class", {}), mix)
                admit_shares = {
                    s: lstats["per_class"].get(s, 0) / max(1, lstats["confident"])
                    for s in mix}
                report["psi_admit_slice"] = round(_psi(admit_shares, mix), 4)
                ok, reason = gates_pass(report, state["baselines"], mix)
                report["gate_reason"] = reason
                # Keep the candidate only on accept. A refused candidate is
                # rolled back, so the live artifact (and its calibration) is
                # byte-identical.
                if ok:
                    report["decision"] = "pr"
                    _refresh_calibration(root, router, report)
                else:
                    _restore_artifact(live, snapshot)
                    report["decision"] = "report-only"
                    report["calibration_refreshed"] = {
                        "status": "not-written",
                        "reason": "the candidate was refused and the previous "
                                  "artifact was restored; the live artifact is "
                                  "byte-identical"}
        except BaseException:
            _restore_artifact(live, snapshot)
            raise
    else:
        report["decision"] = "report-only (do_train=False)"
    _write_pool_state(pool, hard, report, state)
    return report


def _report_path(out_dir: Path, stamp: str) -> Path:
    """The cycle report path, uniquified on collision (-2, -3, ...).

    Two cycles inside one second used to overwrite the earlier report.
    """
    path = out_dir / f"{stamp}.json"
    n = 2
    while path.exists():
        path = out_dir / f"{stamp}-{n}.json"
        n += 1
    return path


def main() -> None:
    import sys

    root = CYCLE_ROOT
    if not (root / "data").is_dir():
        print(
            f"no data/ directory under {Path.cwd()} — a cycle reads its "
            "corpora from the tree it runs in. Run from a root that has "
            "data/ (the repository root, or a scratch copy of it).",
            file=sys.stderr,
        )
        sys.exit(2)
    do_train = "--train" in sys.argv
    retrain_ext = "--all" in sys.argv or "--retrain-extractors" in sys.argv
    report = run_cycle(root=root, do_train=do_train, retrain_extractors=retrain_ext)
    out = root / "reports" / "flywheel"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _report_path(out, stamp).write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "collect"}, indent=2))


if __name__ == "__main__":
    main()
