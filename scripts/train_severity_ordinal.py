#!/usr/bin/env python3
"""Train a CORN/CORAL ordinal severity head on the shipped grade backbone.

The shipped grade artifact (`models/tool-match-grade-int8.onnx`) is a
cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`, the base recorded for
the grade arm in `scripts/train_grade_anchor_alignment.py`) co-trained with a
listwise softmax cross-entropy over ordinal soft targets (MLL,
arXiv:2405.11775). Its residual error on the frozen register panel is
adjacent-tier boundary mass: 33.8% exact on business/SRE phrasing and 26.7% on
adjacent tiers (n=217, `reports/training-report.md`).

This script swaps that softmax CE for a true ordinal loss over the same
backbone family: CORN (Shi et al., arXiv:2111.08851) or CORAL (Cao et al.,
arXiv:1901.07884). Both learn a single latent severity score plus K-1 ordered
cutpoints, so adjacent boundaries share one decision axis instead of competing
for softmax mass.

The head applies to the *fixed* severity scale only; arbitrary user anchors
keep the cross-encoder anchor-alignment path.

Pre-registered bar (stated before measuring): PASS if overall exact >= 50% AND
within-one >= 80% on the frozen panel, OR exact improves by >= 10 points over
the shipped artifact with a paired exact McNemar p < 0.05. FAIL otherwise.

GPU for training; CPU for evaluation; no torch at inference (int8 ONNX +
onnxruntime). Candidate artifact only: writes under data/incoming/models/ and
never promotes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mivia_small_json.determinism import seed_everything  # noqa: E402

BASE = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CORPUS = ROOT / "data" / "real" / "grade"
PANEL = ROOT / "data" / "heldout" / "grade-register-panel.jsonl"
DEFAULT_OUT = ROOT / "data" / "incoming" / "models" / "severity-ordinal-coral-int8.onnx"
K = 4  # INFO, WARNING, ERROR, CRITICAL

# The frozen-panel decode: the exclusive severity anchor templates the shipped
# grade artifact is calibrated on (one source of truth in the bench script).
ANCHOR_TIER = {"info": 0, "warning": 1, "error": 2, "critical": 3}

# Pre-registered bar, printed before any measurement.
# Absolute floors were the promotion bar when shipped was the 39.2%
# cross-encoder (reports/exp-severity-ordinal.md). Since the CORAL head's
# promotion (70.0% panel exact), a candidate on the canonical scale is
# judged against THE SHIPPED HEAD: no panel regression, or a significant
# >= +10pp improvement. The stale floors-alone verdict printed false PASSes
# (found 2026-10-07, tier-balanced distill: 54.8% "PASS" vs shipped 70.0%).
BAR = {"overall_exact": 50.0, "overall_within_one": 80.0, "delta_points": 10.0, "p": 0.05}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _severity(path: Path) -> list[dict]:
    return [r for r in _rows(path)
            if r.get("family") == "severity" and isinstance(r.get("gold"), int)]


def quadratic_weighted_kappa(golds: list[int], preds: list[int], k: int = K) -> float:
    """Cohen's kappa with quadratic weights over an ordinal k-class scale."""
    n = len(golds)
    if n == 0 or k < 2:
        return 0.0
    observed = [[0] * k for _ in range(k)]
    for g, p in zip(golds, preds):
        observed[int(g)][int(p)] += 1
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * observed[i][j]
            den += w * (row[i] * col[j]) / n
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - num / den


def ordinal_metrics(golds: list[int], preds: list[int], k: int = K) -> dict:
    """Exact match, Average Ordinal Distance, within-one, quadratic kappa."""
    n = len(golds)
    if n == 0:
        return {"n": 0, "exact": 0.0, "aod": 0.0, "within_one": 0.0, "qwk": 0.0}
    dist = [abs(int(g) - int(p)) for g, p in zip(golds, preds)]
    return {
        "n": n,
        "exact": 100.0 * sum(d == 0 for d in dist) / n,
        "aod": sum(dist) / n,
        "within_one": 100.0 * sum(d <= 1 for d in dist) / n,
        "qwk": quadratic_weighted_kappa(golds, preds, k),
    }


def coral_decode(cumulative_logits) -> int:
    """Tier index = number of ordered cutpoints the latent exceeds (CORAL)."""
    return sum(1 for z in cumulative_logits if z > 0.0)


def corn_decode(cumulative_logits) -> int:
    """Tier index from the conditional chain: P(y>k) = prod_j sigmoid(z_j)."""
    running = 1.0
    tier = 0
    for z in cumulative_logits:
        running *= 1.0 / (1.0 + math.exp(-float(z)))
        if running > 0.5:
            tier += 1
        else:
            break
    return tier


def soft_cumulative_targets(golds: list[int], k_tiers: int,
                            mass: float) -> list[list[float]]:
    """Diaz & Marathe (CVPR 2019) adjacent-tier soft targets, folded back
    at the scale ends, expressed as CORAL cutpoint targets: target[k] is
    P(tier > k) under the soft tier distribution. Rank-consistent by
    construction; monotone non-increasing in k.

    Boundary rows fold the outside mass back onto the gold tier, so a
    boundary gold keeps mass 1 - mass/2 against its single neighbor.
    """
    if not 0.0 <= mass < 1.0:
        raise ValueError("mass must be in [0, 1)")
    out = []
    for y in golds:
        if not 0 <= y < k_tiers:
            raise ValueError(f"gold {y} outside 0..{k_tiers - 1}")
        dist = [0.0] * k_tiers
        if y == 0:
            dist[0] += 1.0 - mass / 2.0
            if k_tiers > 1:
                dist[1] += mass / 2.0
        elif y == k_tiers - 1:
            dist[y] += 1.0 - mass / 2.0
            dist[y - 1] += mass / 2.0
        else:
            dist[y] += 1.0 - mass
            dist[y - 1] += mass / 2.0
            dist[y + 1] += mass / 2.0
        total = sum(dist)
        dist = [d / total for d in dist]
        # target[k] = P(tier > k)
        out.append([sum(dist[k + 1:]) for k in range(k_tiers - 1)])
    return out


def _decode(logits, loss: str) -> int:
    return corn_decode(logits) if loss == "corn" else coral_decode(logits)


def _shipped_predictions(panel: list[dict]) -> list[int]:
    """The SHIPPED grade path for the canonical severity scale.

    Since the CORAL ordinal head's promotion (reports/exp-severity-ordinal.md,
    panel 70.0% exact), ``models/severity-ordinal-int8.onnx`` is the shipped
    artifact for the fixed 4-tier scale; the cross-encoder anchor matcher
    (39.2%) only serves arbitrary anchor strings. Comparing candidates
    against the legacy matcher here produced false PASS verdicts (found
    2026-10-07: the tier-balanced distill printed PASS at 54.8% while
    regressing the real shipped head 70.0% -> 54.8%).
    """
    from mivia_small_json.tool_match import load_severity_ordinal

    head = load_severity_ordinal()
    if head is None:
        raise SystemExit("shipped severity-ordinal artifact unavailable")
    preds = []
    for row in panel:
        _anchor, _score, tier_set = head.grade(row["text"], with_set=True)
        preds.append(tier_set[0])
    return preds


def main() -> int:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["coral", "corn"], default="coral")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--balance", choices=["none", "inv", "sqrt"], default="inv",
                    help="class-balanced sampling: inverse count or 1/sqrt(count)")
    ap.add_argument("--soft-adjacent", type=float, default=0.0,
                    help="CORAL soft targets (Diaz & Marathe, CVPR 2019): put "
                         "this mass on the gold tier's adjacent tiers (split "
                         "evenly, folded back at the scale ends) before "
                         "deriving the cumulative cutpoint targets. 0 = hard "
                         "targets. CORAL only; ignored for --loss corn.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--extra-train", type=Path, default=None,
                    help="extra train-only jsonl with {text, gold int} rows "
                         "(e.g. an agreed teacher pool); mixed into train, "
                         "never into dev/panel; refused on any eval collision")
    ap.add_argument("--skip-shipped", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=1,
                    help="panel eval batch; 1 matches the shipped per-row path "
                         "(the int8 graph is not batch-invariant)")
    ap.add_argument("--eval-only", type=Path, default=None,
                    help="skip training and evaluate an existing artifact")
    # Arm 3 (security-severity spec): reuse the whole recipe on a different
    # corpus/scale without touching the shipped measurement defaults.
    ap.add_argument("--corpus", type=Path, default=None,
                    help="override the train/dev corpus dir (same jsonl shape)")
    ap.add_argument("--eval-file", type=Path, default=None,
                    help="one-shot eval file (rows with text+gold int); "
                         "replaces the panel comparison (no shipped-vs-cand)")
    ap.add_argument("--skip-shipped-compare", action="store_true")
    args = ap.parse_args()
    global CORPUS, PANEL
    if args.corpus is not None:
        CORPUS = args.corpus if args.corpus.is_absolute() else (ROOT / args.corpus)

    out = args.eval_only if args.eval_only is not None else (
        args.out or DEFAULT_OUT.with_name(DEFAULT_OUT.name.replace("coral", args.loss)))
    out = out if out.is_absolute() else (ROOT / out)

    if args.eval_only is not None:
        if not out.is_file():
            raise SystemExit(f"artifact not found: {out}")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"eval-only: {out.name} ({out.stat().st_size} bytes, "
              f"sha256 {digest[:16]}) · eval batch {args.eval_batch}")
        tok = AutoTokenizer.from_pretrained(BASE)
        return run_panel_eval(out, args, tok, None, digest)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} (cuda available: {torch.cuda.is_available()})")
    print(f"loss: {args.loss.upper()} · backbone: {BASE} · pool: {args.pool} "
          f"· balance: {args.balance}")
    print(f"pre-registered bar: {BAR}")

    train_rows = _severity(CORPUS / "train.jsonl")
    dev_rows = _severity(CORPUS / "dev.jsonl")
    if args.eval_file is not None and (args.eval_file.is_absolute() and False):
        pass  # handled at eval time below
    if args.extra_train is not None:
        import re as _re
        xp = args.extra_train if args.extra_train.is_absolute() \
            else ROOT / args.extra_train
        # NOTE: type() is intentional — isinstance(True, int) is True, so a
        # JSON `true` gold would otherwise slip in as tier 1.
        extra = [{"text": r["text"], "gold": int(r["gold"])} for r in _rows(xp)
                 if isinstance(r.get("text"), str)
                 and type(r.get("gold")) is int and 0 <= r["gold"] < K]
        seen = set()
        extra = [r for r in extra
                 if not (_re.sub(r"\s+", " ", r["text"].strip().lower()) in seen
                         or seen.add(_re.sub(r"\s+", " ", r["text"].strip().lower())))]
        def _norm(text: str) -> str:
            return _re.sub(r"\s+", " ", text.strip().lower())

        frozen_paths = [p for p in (CORPUS / "dev.jsonl", PANEL,
                                    CORPUS / "business_slice.jsonl") if p.is_file()]
        frozen: set[str] = set()
        for path in frozen_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    frozen.add(_norm(json.loads(line)["text"]))
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
        pairs_path = (ROOT / "data" / "incoming" / "external"
                      / "bfcl" / "pairs.jsonl")
        if pairs_path.is_file():
            for line in pairs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    prow = json.loads(line)
                except ValueError:
                    continue
                for key in ("request", "text", "prompt"):
                    if isinstance(prow.get(key), str):
                        frozen.add(_norm(prow[key]))
        clash = [r["text"] for r in extra
                 if _re.sub(r"\s+", " ", r["text"].strip().lower()) in frozen]
        if clash:
            raise SystemExit(f"REFUSED: {len(clash)} extra rows collide "
                             f"with dev/panel/slice/pairs, e.g. {clash[0][:80]!r}")
        train_rows = train_rows + extra
        print(f"extra train rows: {len(extra)} from {xp.name} "
              f"(histogram {dict(sorted(Counter(r['gold'] for r in extra).items()))})")
    print(f"severity rows: {len(train_rows)} train / {len(dev_rows)} dev")
    print(f"train tier histogram: "
          f"{dict(sorted(Counter(r['gold'] for r in train_rows).items()))}")

    tok = AutoTokenizer.from_pretrained(BASE)

    class OrdinalSeverity(nn.Module):
        """Shared latent + K-1 ordered cutpoints (CORAL/CORN)."""

        def __init__(self, base_name: str, k: int, pool: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_name)
            self.pool = pool
            hidden = self.encoder.config.hidden_size
            self.latent = nn.Linear(hidden, 1)
            self.cut0 = nn.Parameter(torch.zeros(1))
            self.deltas = nn.Parameter(torch.zeros(max(0, k - 2)))

        def cutpoints(self):
            if self.deltas.numel() == 0:
                return self.cut0
            return torch.cat([
                self.cut0,
                self.cut0 + torch.cumsum(F.softplus(self.deltas), dim=0),
            ])

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            if self.pool == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            latent = self.latent(pooled).squeeze(-1)
            return latent.unsqueeze(-1) - self.cutpoints().unsqueeze(0)

    def coral_loss(logits, golds):
        # Hard: 1{gold > k}. Soft (--soft-adjacent): cumulative targets are
        # P(tier > k) under the Diaz & Marathe adjacent-tier distribution.
        idx = torch.arange(K - 1, device=logits.device).unsqueeze(0)
        if getattr(args, "soft_adjacent", 0.0) <= 0.0:
            targets = (golds.unsqueeze(1) > idx).float()
        else:
            soft = soft_cumulative_targets(golds.tolist(), K, args.soft_adjacent)
            targets = torch.tensor(soft, device=logits.device,
                                   dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, targets)

    def corn_loss(logits, golds):
        # Task k predicts P(y > k | y > k-1) on the subset {y >= k}.
        total = logits.new_zeros(())
        tasks = 0
        for k in range(K - 1):
            cond = golds >= k
            if not bool(cond.any()):
                continue
            target = (golds[cond] > k).float()
            total = total + F.binary_cross_entropy_with_logits(logits[cond, k], target)
            tasks += 1
        return total / max(1, tasks)

    loss_fn = coral_loss if args.loss == "coral" else corn_loss

    seed_everything(args.seed)
    net = OrdinalSeverity(BASE, K, args.pool).to(device)

    def make_sampler():
        if args.balance == "none":
            return None
        counts = Counter(r["gold"] for r in train_rows)
        expo = 1.0 if args.balance == "inv" else 0.5
        weights = [1.0 / (counts[r["gold"]] ** expo) for r in train_rows]
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(weights, num_samples=len(train_rows),
                                     replacement=True)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    steps_per_epoch = max(1, (len(train_rows) + args.batch - 1) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def batch_indices():
        sampler = make_sampler()
        if sampler is None:
            return torch.randperm(len(train_rows)).tolist()
        return list(sampler)

    def evaluate(rows) -> tuple[float, list[int]]:
        net.eval()
        preds: list[int] = []
        for start in range(0, len(rows), 64):
            chunk = rows[start:start + 64]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = net(enc["input_ids"], enc["attention_mask"]).cpu().tolist()
            preds.extend(_decode(row, args.loss) for row in logits)
        correct = sum(p == r["gold"] for p, r in zip(preds, rows))
        return (100.0 * correct / len(rows) if rows else 0.0), preds

    step = 0
    for epoch in range(args.epochs):
        net.train()
        perm = batch_indices()
        total = 0.0
        batches = 0
        for start in range(0, len(perm), args.batch):
            chunk = [train_rows[i] for i in perm[start:start + args.batch]]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            golds = torch.tensor([r["gold"] for r in chunk], device=device)
            logits = net(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(logits, golds)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            total += float(loss.detach())
            batches += 1
        dev_acc, _ = evaluate(dev_rows)
        print(f"epoch {epoch + 1}: loss {total / max(1, batches):.4f} · "
              f"corpus dev exact {dev_acc:.1f}%")

    dev_acc, dev_preds = evaluate(dev_rows)
    print(f"corpus dev exact: {dev_acc:.1f}%")
    print("dev pred histogram:",
          dict(sorted(Counter(dev_preds).items())))
    print("dev gold histogram:",
          dict(sorted(Counter(r["gold"] for r in dev_rows).items())))
    print("cutpoints:", [round(x, 3) for x in
                         net.cutpoints().detach().cpu().tolist()])

    # Export fp32 + int8.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    net.to("cpu").eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out.parent / out.name.replace("-int8", "")
    dummy = tok(["a log line"], return_tensors="pt", truncation=True,
                max_length=args.max_len, padding=True)
    torch.onnx.export(
        net,
        (dummy["input_ids"], dummy["attention_mask"]),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["cumulative_logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "cumulative_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / (out.name + ".sha256")).write_text(digest + "\n", encoding="utf-8")
    print(f"exported {out.name} ({out.stat().st_size} bytes, sha256 {digest[:16]})")

    # Evaluate on the frozen panel: candidate (CPU ONNX) vs shipped.
    return run_panel_eval(out, args, tok, dev_acc, digest)


def run_panel_eval(out: Path, args, tok, dev_acc, digest) -> int:
    """Evaluate one candidate artifact on the frozen panel against the shipped one.

    Returns the process exit code (0 on the pre-registered PASS, else 1). The
    shipped grade path scores one row per model call, so ``--eval-batch 1`` is
    the honest comparison: the int8 graph is not batch-invariant.
    """
    import onnxruntime as ort

    from mivia_small_json.stats import mcnemar_exact

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    if getattr(args, "eval_file", None):
        panel = _rows(args.eval_file if args.eval_file.is_absolute()
                      else ROOT / args.eval_file)
    else:
        panel = _rows(PANEL)
    batch = args.eval_batch

    def candidate_predictions() -> list[int]:
        preds: list[int] = []
        for start in range(0, len(panel), batch):
            chunk = panel[start:start + batch]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True)
            logits = sess.run(["cumulative_logits"], {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]})[0]
            preds.extend(_decode(row.tolist(), args.loss) for row in logits)
        return preds

    cand = candidate_predictions()
    shipped = (None if args.skip_shipped or args.eval_file
               else _shipped_predictions(panel))
    print(f"\n== frozen panel (eval batch {batch}) ==")
    print(f"{'stratum':18s} {'n':>4s} | {'shipped exact':>13s} | "
          f"{'cand exact':>10s} {'aod':>5s} {'within1':>7s} {'qwk':>5s} | "
          f"{'mcnemar p':>9s}")
    table: dict[str, dict] = {}
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(panel):
        groups.setdefault(row.get("stratum", "all"), []).append(i)
    groups["ALL"] = list(range(len(panel)))

    for label in sorted(groups, key=lambda s: (s == "ALL", s)):
        idxs = groups[label]
        g = [panel[i]["gold"] for i in idxs]
        c = [cand[i] for i in idxs]
        cm = ordinal_metrics(g, c)
        entry = {"candidate": cm}
        if shipped is not None:
            s = [shipped[i] for i in idxs]
            sm = ordinal_metrics(g, s)
            b = sum(s[j] == g[j] and c[j] != g[j] for j in range(len(idxs)))
            cc = sum(c[j] == g[j] and s[j] != g[j] for j in range(len(idxs)))
            entry["shipped"] = sm
            entry["mcnemar_p"] = mcnemar_exact(b, cc)
            print(f"{label:18s} {cm['n']:4d} | {sm['exact']:12.1f}% | "
                  f"{cm['exact']:9.1f}% {cm['aod']:5.2f} {cm['within_one']:6.1f}% "
                  f"{cm['qwk']:5.3f} | {entry['mcnemar_p']:9.4f}")
        else:
            print(f"{label:18s} {cm['n']:4d} | {'-':>13s} | "
                  f"{cm['exact']:9.1f}% {cm['aod']:5.2f} {cm['within_one']:6.1f}% "
                  f"{cm['qwk']:5.3f} | {'-':>9s}")
        table[label] = entry

    # Pre-registered verdict.
    overall = table["ALL"]
    cand_exact = overall["candidate"]["exact"]
    cand_within = overall["candidate"]["within_one"]
    passed = cand_exact >= BAR["overall_exact"] and cand_within >= BAR["overall_within_one"]
    delta = None
    p_value = None
    if "shipped" in overall:
        delta = cand_exact - overall["shipped"]["exact"]
        p_value = overall["mcnemar_p"]
        # Shipped is the CORAL head now: absolute floors alone do not pass.
        # A candidate must not regress the panel, or must improve it by
        # >= BAR["delta_points"] with significance.
        not_regressing = (delta >= 0.0
                          and cand_within >= overall["shipped"]["within_one"])
        significant_gain = (delta >= BAR["delta_points"]
                            and p_value < BAR["p"])
        passed = not_regressing or significant_gain
    print(f"\ncandidate overall exact {cand_exact:.1f}% · within-one "
          f"{cand_within:.1f}%"
          + (f" · delta {delta:+.1f}pp (p={p_value:.4f})" if delta is not None else ""))
    print(f"verdict: {'PASS' if passed else 'FAIL'}")

    # An eval-only run must not clobber the training metadata a train+export run
    # wrote: carry the training fields over from the existing sidecar.
    prior: dict = {}
    sidecar = out.parent / (out.name + ".results.json")
    if args.eval_only is not None and sidecar.is_file():
        try:
            prior = json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            prior = {}

    results = {
        "loss": args.loss,
        "base": BASE,
        "pool": prior.get("pool", args.pool),
        "balance": prior.get("balance", args.balance),
        "epochs": prior.get("epochs", args.epochs),
        "seed": prior.get("seed", args.seed),
        "eval_batch": batch,
        "artifact": str(out.relative_to(ROOT)),
        "artifact_bytes": out.stat().st_size,
        "artifact_sha256": digest,
        "corpus_dev_exact": dev_acc if dev_acc is not None
                            else prior.get("corpus_dev_exact"),
        "bar": BAR,
        "delta_points": delta,
        "mcnemar_p": p_value,
        "verdict": "PASS" if passed else "FAIL",
        "panel": table,
    }
    sidecar.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.name}.results.json")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
