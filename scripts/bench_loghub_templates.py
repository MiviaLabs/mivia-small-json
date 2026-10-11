#!/usr/bin/env python3
"""LogHub template-parsing sub-benchmark: GA / PA / PA-matched / FGA / FTA.

Runs the in-repo Drain parser (``mivia_small_json.logtemplate``) over the
message column of every LogHub system and scores it against the vendored
ground truth (``<System>_2k.log_structured.csv``: ``EventId`` and
``EventTemplate``). Metrics follow LogPai's ``logparser`` evaluator so the
numbers sit beside published Drain / Spell / IPLoM results. Alongside the
strict string-equality PA, the script computes a cluster-matched companion
(``parse_accuracy_matched`` below) that credits each predicted cluster with
its majority-overlap ground-truth template.

The original Loghub-2k ground truth contains wrong event templates (Le &
Zhang, *Guidelines for Assessing the Accuracy of Log Message Template
Identification*, ICSE 2022). ``--corrected`` scores against the upstream
corrected labels (``logpai/logparser`` ``data/loghub_2k_corrected/``,
vendored under ``data/incoming/loghub-upstream-corrected/``) when present,
falling back to the original GT per system. ``--compare`` scores both and
writes a before/after report.

This complements ``scripts/bench_loghub.py``, which scores the pipeline's
routing and field extraction, not template parsing.

Writes reports/benchmark-loghub-templates.md. Deterministic, offline.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.logtemplate import Drain, evaluate  # noqa: E402

LOGHUB = ROOT / "data" / "incoming" / "loghub-upstream"
LOGHUB_CORRECTED = ROOT / "data" / "incoming" / "loghub-upstream-corrected"
REPORT = ROOT / "reports" / "benchmark-loghub-templates.md"

# Published Drain average parsing accuracy (PA) on Loghub-2k, from Zhu et
# al., "Tools and Benchmarks for Automated Log Parsing", ICSE 2019,
# arXiv:1811.03509, Table IV (last row: Drain 0.865). The GA/FGA/FTA
# protocol itself comes from Loghub-2.0 (arXiv:2308.10828).
PUBLISHED_DRAIN_PA = 0.865

# Pre-registered reading: a corrected-GT PA shift larger than this means the
# original PA was partly ground-truth label noise, and the corrected number
# is the honest baseline.
PA_NOISE_THRESHOLD = 0.05


def _original_path(system: str) -> Path:
    return LOGHUB / system / f"{system}_2k.log_structured.csv"


def _corrected_path(system: str) -> Path:
    return LOGHUB_CORRECTED / system / f"{system}_2k.log_structured_corrected.csv"


def _systems() -> list[str]:
    """Systems with a structured GT in either the original or corrected tree."""
    names: set[str] = set()
    if LOGHUB.is_dir():
        names |= {
            name for name in os.listdir(LOGHUB)
            if _original_path(name).is_file()
        }
    if LOGHUB_CORRECTED.is_dir():
        names |= {
            name for name in os.listdir(LOGHUB_CORRECTED)
            if _corrected_path(name).is_file()
        }
    return sorted(names)


def _gt_path(system: str, corrected: bool) -> Path:
    """The corrected GT when requested and present, else the original."""
    if corrected and _corrected_path(system).is_file():
        return _corrected_path(system)
    return _original_path(system)


def _rows(system: str, corrected: bool = False) -> list[tuple[str, str, str]]:
    """Ordered ``(content, event_id, template)``, dropping blank ids."""
    path = _gt_path(system, corrected)
    out: list[tuple[str, str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            eid = (row.get("EventId") or "").strip()
            if not eid:
                continue
            out.append(((row.get("Content") or "").strip(), eid,
                        (row.get("EventTemplate") or "").strip()))
    return out


def parse_accuracy_matched(gt_templates: list[str],
                           pred_groupids: list) -> float:
    """Cluster-matched (aligned) PA — the published-Drain-comparable companion.

    Alignment rule, exactly: partition the line indices by predicted cluster
    id. For each predicted cluster, find the ground-truth template string
    with which the cluster shares the most lines (the majority-overlap GT
    template; ties broken by lexicographically smallest template string, so
    the rule is deterministic). Every line in the cluster whose GT template
    equals that matched template is credited. The score is credited lines /
    total lines. Equivalently: clustering purity with GT template strings as
    the reference classes. Strict PA is untouched — it stays the fraction of
    lines whose predicted template string is byte-identical to the GT
    string, and this function never feeds back into it or into the parser.
    """
    if not gt_templates:
        return 0.0
    members: dict = {}
    for i, gid in enumerate(pred_groupids):
        members.setdefault(gid, []).append(i)
    credited = 0
    for idxs in members.values():
        counts = Counter(gt_templates[i] for i in idxs)
        matched = min(counts, key=lambda t: (-counts[t], t))
        credited += counts[matched]
    return credited / len(gt_templates)


def score_system(system: str, corrected: bool = False) -> tuple[dict, list, list, list, list]:
    """Run Drain over one system; return metrics plus the raw label lists."""
    rows = _rows(system, corrected)
    drain = Drain()
    gt_eids, pred_gids, gt_tpls, pred_tpls = [], [], [], []
    for content, eid, template in rows:
        gid, pred_tpl = drain.add(content)
        gt_eids.append(eid)
        pred_gids.append(gid)
        gt_tpls.append(template)
        pred_tpls.append(pred_tpl)
    metrics = evaluate(gt_eids, pred_gids, gt_tpls, pred_tpls)
    metrics["PA_matched"] = round(
        parse_accuracy_matched(gt_tpls, pred_gids), 4)
    return (metrics, gt_eids, pred_gids, gt_tpls, pred_tpls)


def score_all(corrected: bool) -> tuple[dict, dict, int]:
    """Score every system; return ``(per_system, overall, total_lines)``."""
    per_system: dict[str, tuple[int, dict]] = {}
    all_gt_eids: list = []
    all_pred_gids: list = []
    all_gt_tpls: list = []
    all_pred_tpls: list = []
    for system in _systems():
        metrics, gt_e, pr_g, gt_t, pr_t = score_system(system, corrected)
        per_system[system] = (len(gt_e), metrics)
        # Scope group ids by system: raw ids collide across systems.
        all_gt_eids += [f"{system}:{e}" for e in gt_e]
        all_pred_gids += [f"{system}:{g}" for g in pr_g]
        all_gt_tpls += gt_t
        all_pred_tpls += pr_t
    overall = evaluate(all_gt_eids, all_pred_gids, all_gt_tpls, all_pred_tpls)
    overall["PA_matched"] = round(
        parse_accuracy_matched(all_gt_tpls, all_pred_gids), 4)
    return per_system, overall, len(all_gt_eids)


def _delta(new: float, old: float) -> str:
    return f"{new - old:+.4f}"


def _metric_table(per_system: dict, overall: dict, total: int) -> list[str]:
    lines = [
        "| System | Lines | GA | PA | PA-matched | FGA | FTA |",
        "|---|---|---|---|---|---|---|",
    ]
    for system, (count, m) in per_system.items():
        lines.append(
            f"| {system} | {count} | {m['GA']:.4f} | {m['PA']:.4f} | "
            f"{m['PA_matched']:.4f} | {m['FGA']:.4f} | {m['FTA']:.4f} |"
        )
    lines.append(
        f"| **OVERALL** | **{total}** | **{overall['GA']:.4f}** | "
        f"**{overall['PA']:.4f}** | **{overall['PA_matched']:.4f}** | "
        f"**{overall['FGA']:.4f}** | "
        f"**{overall['FTA']:.4f}** |"
    )
    return lines


def _pa_note_lines() -> list[str]:
    return [
        "### Strict PA vs matched PA",
        "",
        "**Strict PA** counts a line correct only when the predicted",
        "template string is byte-identical to the ground-truth template",
        "string — it therefore measures the parser *and* the benchmark's",
        "masking conventions at once: GT writes `blk_<*>` where this parser",
        "keeps `blk_<id>` (HDFS: PA 0.0000 at GA 0.9975), renders",
        "`Invalid user <*>` where the parser keeps the literal username",
        "(OpenSSH), deletes parenthesised constants, and carries whitespace",
        "quirks. **PA-matched** is the cluster-aligned companion: each",
        "predicted cluster is credited with its majority-overlap GT template",
        "(exact rule in `parse_accuracy_matched`), so a line is correct when",
        "it sits in the right group under the right GT template regardless",
        "of string-convention disagreements — this is the protocol closest",
        "to the published-Drain comparison and it isolates grouping quality",
        "from template-string style. Both are reported; neither replaces the",
        "other.",
        "",
    ]


def _reference_lines() -> list[str]:
    return [
        "## Published reference (context only)",
        "",
        "Drain's average parsing accuracy on Loghub-2k is **"
        f"{PUBLISHED_DRAIN_PA}** (Zhu et al., *Tools and Benchmarks for",
        "Automated Log Parsing*, ICSE 2019, arXiv:1811.03509, Table IV). That",
        "figure is message-level grouping correctness, closest to this",
        "report's GA; the strict template-equality PA below is a harder",
        "measure. The GA/FGA/FTA protocol is defined in Loghub-2.0",
        "(arXiv:2308.10828). Per-system numbers depend on parser",
        "configuration; treat this as a sanity check, not a leaderboard entry.",
        "",
    ]


def _comparison_lines(orig_sys: dict, orig_overall: dict, orig_total: int,
                      corr_sys: dict, corr_overall: dict, corr_total: int,
                      date: str) -> list[str]:
    lines = [
        f"## Corrected ground truth ({date})",
        "",
        "Loghub-2k's original GT carries wrong event templates (Le & Zhang,",
        "*Guidelines for Assessing the Accuracy of Log Message Template",
        "Identification*, ICSE 2022). Re-scored against the upstream corrected",
        "labels (`logpai/logparser` `data/loghub_2k_corrected/`, vendored under",
        "`data/incoming/loghub-upstream-corrected/`). The message column is",
        "unchanged; only `EventId` / `EventTemplate` differ.",
        "",
        "| System | GA orig | GA corr | ΔGA | PA orig | PA corr | ΔPA | "
        "PA-m orig | PA-m corr | ΔPA-m | "
        "FGA orig | FGA corr | ΔFGA | FTA orig | FTA corr | ΔFTA |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for system, (_, o) in orig_sys.items():
        c = corr_sys[system][1]
        lines.append(
            f"| {system} | {o['GA']:.4f} | {c['GA']:.4f} | "
            f"{_delta(c['GA'], o['GA'])} | {o['PA']:.4f} | {c['PA']:.4f} | "
            f"{_delta(c['PA'], o['PA'])} | {o['PA_matched']:.4f} | "
            f"{c['PA_matched']:.4f} | {_delta(c['PA_matched'], o['PA_matched'])} | "
            f"{o['FGA']:.4f} | {c['FGA']:.4f} | "
            f"{_delta(c['FGA'], o['FGA'])} | {o['FTA']:.4f} | {c['FTA']:.4f} | "
            f"{_delta(c['FTA'], o['FTA'])} |"
        )
    lines.append(
        f"| **OVERALL** | **{orig_overall['GA']:.4f}** | "
        f"**{corr_overall['GA']:.4f}** | "
        f"**{_delta(corr_overall['GA'], orig_overall['GA'])}** | "
        f"**{orig_overall['PA']:.4f}** | **{corr_overall['PA']:.4f}** | "
        f"**{_delta(corr_overall['PA'], orig_overall['PA'])}** | "
        f"**{orig_overall['PA_matched']:.4f}** | "
        f"**{corr_overall['PA_matched']:.4f}** | "
        f"**{_delta(corr_overall['PA_matched'], orig_overall['PA_matched'])}** | "
        f"**{orig_overall['FGA']:.4f}** | **{corr_overall['FGA']:.4f}** | "
        f"**{_delta(corr_overall['FGA'], orig_overall['FGA'])}** | "
        f"**{orig_overall['FTA']:.4f}** | **{corr_overall['FTA']:.4f}** | "
        f"**{_delta(corr_overall['FTA'], orig_overall['FTA'])}** |"
    )
    dpa = corr_overall["PA"] - orig_overall["PA"]
    if abs(dpa) > PA_NOISE_THRESHOLD:
        verdict = (
            f"**Verdict:** corrected-GT PA moves {dpa:+.4f} "
            f"(|Δ| > {PA_NOISE_THRESHOLD}), so the original PA "
            f"{orig_overall['PA']:.4f} was partly ground-truth label noise; "
            f"the corrected {corr_overall['PA']:.4f} is the honest baseline."
        )
    else:
        verdict = (
            f"**Verdict:** corrected-GT PA moves {dpa:+.4f} "
            f"(|Δ| ≤ {PA_NOISE_THRESHOLD}), so the original PA "
            f"{orig_overall['PA']:.4f} was not dominated by ground-truth "
            f"label noise; the corrected {corr_overall['PA']:.4f} is the "
            f"honest baseline."
        )
    return lines + ["", verdict, ""]


def _fallback_systems() -> list[str]:
    return [s for s in _systems() if not _corrected_path(s).is_file()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--corrected", action="store_true",
        help="score against the corrected Loghub-2k GT when present, "
             "falling back to the original GT per system",
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="score both GTs and write the before/after report section",
    )
    args = ap.parse_args(argv)

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if args.compare:
        orig_sys, orig_overall, orig_total = score_all(False)
        corr_sys, corr_overall, corr_total = score_all(True)
        lines = [
            "# LogHub template-parsing benchmark (GA / PA / PA-matched / FGA / FTA)",
            "",
            "In-repo Drain parser (`src/mivia_small_json/logtemplate.py`) over the",
            "message column of each system, scored against the vendored ground",
            "truth (`_2k.log_structured.csv`: EventId + EventTemplate). Metrics",
            "follow LogPai's `logparser` evaluator.",
            "",
            f"Run: {date} · script: `scripts/bench_loghub_templates.py` "
            "(`--compare`)",
            "",
        ]
        lines += _metric_table(orig_sys, orig_overall, orig_total)
        lines += [""]
        lines += _pa_note_lines()
        lines += _comparison_lines(orig_sys, orig_overall, orig_total,
                                   corr_sys, corr_overall, corr_total, date)
        lines += _reference_lines()
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines), encoding="utf-8")
        print(f"original  overall: {orig_overall}")
        print(f"corrected overall: {corr_overall}")
        print(f"report written to {REPORT.relative_to(ROOT)}")
        return 0

    per_system, overall, total = score_all(args.corrected)
    gt_label = "corrected" if args.corrected else "original"
    lines = [
        "# LogHub template-parsing benchmark (GA / PA / PA-matched / FGA / FTA)",
        "",
        "In-repo Drain parser (`src/mivia_small_json/logtemplate.py`) over the",
        "message column of each system, scored against the vendored ground",
        "truth (`_2k.log_structured.csv`: EventId + EventTemplate). Metrics",
        "follow LogPai's `logparser` evaluator.",
        "",
        f"Run: {date} · script: `scripts/bench_loghub_templates.py` "
        f"· GT: {gt_label}",
        "",
    ]
    if args.corrected:
        missing = _fallback_systems()
        note = ("Scored against the corrected GT where present "
                "(`data/incoming/loghub-upstream-corrected/`).")
        if missing:
            note += " Fell back to the original GT for: " + ", ".join(missing) + "."
        else:
            note += " All systems had a corrected GT."
        lines += [note, ""]
    lines += _metric_table(per_system, overall, total)
    lines += [""]
    lines += _pa_note_lines()
    lines += _reference_lines()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"overall ({gt_label} GT): {overall}")
    print(f"report written to {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
