#!/usr/bin/env python3
"""Build the cross-encoder's (request, tool, label) pair corpus.

The guard answers "does this request match a tool in this catalogue?".
Training it from single-tool catalogues alone taught it to agree with any
listed tool; training it from random foreign tools taught it that a listed
tool matches whenever the request's values fit the tool's parameters. The
second failure is structural alignment bias: near-miss catalogues pair a
request with a topically adjacent tool whose parameters accept the request's
attributes but whose function cannot serve it. The eval is built from exactly
those pairs, so the corpus must contain them.

Four negative sources:

- **within-row distractors**: the listed tools that are not the answer
- **aligned distractors**: the within-row distractors whose *required-parameter
  signature equals a gold tool's*. These are BFCL's own
  structurally-aligned-but-irrelevant labels: same interface, different
  function (`country_info.largest_city` against a capital-city request)
- **foreign tools**: the same request paired with a tool from another row,
  so every request has both a positive and a negative and the two classes
  share one length distribution by construction
- **the 236 irrelevance rows**: real BFCL near-misses

`--nearmiss {xlam,when2call,both}` augments the corpus with external synthetic
near-misses (Hammer, arXiv 2410.04587; both sources CC BY 4.0): the
`xlam-irrelevance-7.5k` gold-tool-removed catalogues (foreign-tool negatives)
and the When2Call MCQ sibling lists (aligned-but-irrelevant negatives). They
add near-miss evidence the BFCL pool cannot, without the same-signature mining
the next paragraph rules out. Measured and falsified as a guard lever:
`reports/exp-nearmiss-augmentation.md`.

The shipped corpus is the **v1 span** (name and description), which is what the
shipped guard reads. `--rendering v2` appends the parameter schema; it raised
abstention but cost retention, so no model trained with it ships. `--sabeval`
mixes in the SABEval sibling pairs (ACL 2026, `along-l/irrelevant-tool`): 101
tool templates whose sibling classes share one interface but have mutually
exclusive functionality. It is learned in-domain but does not transfer to
BFCL, so it is off by default; the flag reproduces the falsified arms.

Mining same-signature foreign tools from BFCL's own pool is deliberately NOT
done: the pool re-declares the same function under different names across
splits (`gcd`/`math.gcd` at description cosine 1.0; `math.hcf`/`gcd` synonyms
at 0.18), so a similarity cut cannot separate synonyms from different
functions and would teach false negatives. Only labels BFCL itself asserts
(distractors) or that SABEval guarantees by construction are used.

Ground truth for BFCL comes from `possible_answer/BFCL_v4_<split>.json`.

The corpus is built to be disjoint from both eval sets:

- `data/incoming/external/bfcl/calls.jsonl` (tool-call retention)
- `data/incoming/external/bfcl/irrelevance_eval.jsonl` (abstention)

Both are scored end to end, so every request they contain is dropped here,
along with every pair that request forms. The filter is at the request level,
not the pair level: BFCL reuses a question across splits, so dropping only the
eval row's exact pair would leave the same request's other pairs in the corpus
as negatives and teach the model to reject it. Without that, the reported
numbers are measured on requests the guard trained on.

Writes `data/incoming/external/bfcl/pairs.jsonl`. Deterministic, offline after
the first fetch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mivia_small_json.tool_match import (  # noqa: E402
    parse_catalogue_raw,
    render_tool,
    render_tool_fillability,
    validate_fillability,
)

EXT = ROOT / "data" / "incoming" / "external"
BFCL = EXT / "bfcl"
OUT = BFCL / "pairs.jsonl"
# The abstention eval set. `live_irrelevance` is 884 rows disjoint from the 240
# in `irrelevance.jsonl`, so the 240 can train the guard while the 884 stay
# unseen. The 240 alone were too few near-miss negatives to hold precision.
FP_EVAL = BFCL / "irrelevance_eval.jsonl"
SPLITS = ("simple_python", "simple_java", "simple_javascript", "parallel",
          "multiple", "parallel_multiple",
          # The `live_*` splits are the terse, natural-request style. The
          # end-to-end eval sets are drawn from that style, and a corpus of
          # only the narrative splits above left the model scoring an obvious
          # terse match ("How many ounces in 2 pounds of butter?" against
          # cooking_conversion.convert) at 0.003. `live_relevance` is excluded:
          # it is the source of the retention eval set and ships no
          # possible_answer file, because it labels relevance, not calls.
          "live_simple", "live_multiple", "live_parallel",
          "live_parallel_multiple")
BASE = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
        "berkeley-function-call-leaderboard/bfcl_eval/data")
# SABEval: pinned so the corpus is reproducible. Commit 6f41fe97 is the
# verified 2026-07-03 HEAD of along-l/irrelevant-tool (ACL 2026). The repo
# ships no license file; license verification is deferred by the owner
# decision of 2026-10-05. See HARDER_SETS_PROVENANCE.md.
SABEVAL_COMMIT = "6f41fe970ca14d2699447e3b0c184b77afc45eb7"
SABEVAL_URL = (f"https://github.com/along-l/irrelevant-tool/archive/"
               f"{SABEVAL_COMMIT}.tar.gz")
SABEVAL_TAR = Path(f"/tmp/sabeval-{SABEVAL_COMMIT}.tar.gz")
SABEVAL_DIR = Path(f"/tmp/sabeval-{SABEVAL_COMMIT}")
# External synthetic near-miss augmentation (Hammer, arXiv 2410.04587). Both
# sets are CC BY 4.0. `xlam-irrelevance-7.5k` removes the ground-truth function
# from a real xlam-function-calling catalogue and relabels the row irrelevant;
# When2Call's MCQ `cannot_answer` rows list sibling tools that cannot serve the
# question. Neither is BFCL, so they add near-miss evidence the BFCL pool
# cannot (BFCL's own same-signature mining is deliberately not done; see the
# module docstring). Caps keep the negative class from swamping the positives.
XLAM_REPO = "MadeAgents/xlam-irrelevance-7.5k"
XLAM_FILE = "xlam-7.5k-irrelevancek.json"
XLAM_DIR = EXT / "xlam-irrelevance"
XLAM_CAP = 6000
WHEN2CALL_REPO = "nvidia/When2Call"
WHEN2CALL_MCQ = "test/when2call_test_mcq.jsonl"
WHEN2CALL_DIR = EXT / "when2call"
WHEN2CALL_CAP = 3000
# BFCL multi-turn (Apache-2.0): the session-conditioned evidence source.
# ``multi_turn_miss_func`` excludes the function a turn needs, so that turn's
# catalogue is all-negative (an abstention signal in session form);
# ``multi_turn_miss_param`` keeps the function but strips a required argument;
# ``multi_turn_base`` has neither. The per-turn ground truth is in
# ``possible_answer/``; the tool schemas are one JSONL file per class under
# ``multi_turn_func_doc/``. ``path`` is the full gold trajectory and is NOT
# aligned 1:1 with ``question`` turns, so labels come from the per-turn
# ``ground_truth`` lists, never from ``path``.
MT_SPLITS = ("multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param")
MT_DEFAULT = "multi_turn_miss_func"
MT_CLASS_FILES = {
    "GorillaFileSystem": "gorilla_file_system",
    "VehicleControlAPI": "vehicle_control",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "MathAPI": "math_api",
}
MT_DOC_DIR = Path("/tmp/bfcl_mt_func_doc")
# The session channel appended to a tool span. ``(none)`` is the single-turn
# case, and is what every row of the two end-to-end eval sets renders to.
SESSION_NONE = "(none)"
MT_PRIOR_TURNS_CHARS = 240
MT_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
# Unicode line/paragraph separators are valid inside a JSON string but
# ``str.splitlines()`` splits on them, so a line-splitting reader breaks.
_SEPARATORS = re.compile("[\u2028\u2029\x85\x1c\x1d\x1e]")


def _clean(text: str) -> str:
    return _SEPARATORS.sub(" ", text).strip()


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _split_of(request: str, tool: str) -> str:
    """Deterministic 80/20 split, keyed on the pair itself."""
    return "dev" if int(_digest(request + tool), 16) % 5 == 0 else "train"


def _fetch(url: str, cached: Path) -> list[dict]:
    if not cached.is_file():
        req = urllib.request.Request(
            url, headers={"User-Agent": "mivia-small-json/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(resp.read())
        except urllib.error.HTTPError as exc:
            print(f"skipping {url.rsplit('/', 1)[-1]}: HTTP {exc.code}",
                  file=sys.stderr)
            return []
    # BFCL ships one JSON object per line, not a JSON array.
    out = []
    for line in cached.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            out.append(json.loads(line))
    return out


def _sources() -> tuple[list[dict], dict[str, dict]]:
    questions, answers = [], {}
    for split in SPLITS:
        q = _fetch(f"{BASE}/BFCL_v4_{split}.json",
                   Path(f"/tmp/bfcl_v4_{split}.json"))
        a = _fetch(f"{BASE}/possible_answer/BFCL_v4_{split}.json",
                   Path(f"/tmp/bfcl_v4_{split}_answer.json"))
        questions += q
        answers.update({r["id"]: r["ground_truth"] for r in a})
    return questions, answers


def _fp_eval_rows() -> list[dict]:
    """The abstention eval set: BFCL `live_irrelevance` (884 rows)."""
    return _fetch(f"{BASE}/BFCL_v4_live_irrelevance.json",
                  Path("/tmp/bfcl_v4_live_irrelevance.json"))


def _eval_requests() -> set[str]:
    """Every request either end-to-end eval set scores.

    The filter is at the *request* level, not the pair level. BFCL reuses a
    question across splits, so dropping only the exact (request, tool) pair an
    eval row carries leaves that request's other pairs in the corpus — almost
    always as negatives. The model then learns "this request needs no tool",
    which understates retention on the 178 `calls.jsonl` requests and
    overstates abstention on the 766 `live_irrelevance` ones. Dropping the
    request makes the eval rows genuinely unseen.

    `calls.jsonl` scores retention and `live_irrelevance` scores abstention.
    The 240 rows in `irrelevance.jsonl` are deliberately absent: the guard
    trains its abstention signal on them.
    """
    blocked: set[str] = set()
    path = BFCL / "calls.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            _tools, request = parse_catalogue_raw(
                json.loads(line).get("text") or "")
            blocked.add(_clean(request))
    for rec in _fp_eval_rows():
        blocked.add(_clean(rec["question"][0][0]["content"]))
    return blocked


def _serialize(rec: dict) -> str:
    return ("Tools available:\n" + json.dumps(rec["function"]) +
            "\n\nUser request: " + rec["question"][0][0]["content"])


def _write_fp_eval(rows: list[dict]) -> int:
    payload = "".join(
        json.dumps({"source": "bfcl-live-irrelevance",
                    "text": _clean(_serialize(rec))}, sort_keys=True) + "\n"
        for rec in rows)
    FP_EVAL.write_text(payload, encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------------
# BFCL: aligned (same-signature) distractor tagging
# ---------------------------------------------------------------------------
def _signature(tool: dict) -> tuple:
    """The tool's required (name, type) parameters, or all of them."""
    params = tool.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = required if isinstance(required, list) else []
    keys = [k for k in required if k in props] or list(props)
    return tuple(sorted((k, str((props.get(k) or {}).get("type", "")))
                        for k in keys))


# ---------------------------------------------------------------------------
# SABEval: sibling tools share one interface, differ in function
# ---------------------------------------------------------------------------
def _sabeval_root() -> Path | None:
    """The SABEval data directory from the pinned commit, or None."""
    if not SABEVAL_DIR.is_dir():
        if not SABEVAL_TAR.is_file():
            req = urllib.request.Request(
                SABEVAL_URL, headers={"User-Agent": "mivia-small-json/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    SABEVAL_TAR.write_bytes(resp.read())
            except urllib.error.HTTPError as exc:
                print(f"skipping SABEval: HTTP {exc.code}", file=sys.stderr)
                return None
        SABEVAL_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(SABEVAL_TAR) as tar:
            tar.extractall(SABEVAL_DIR, filter="data")
    hits = list(SABEVAL_DIR.glob("*/data/SABEval"))
    return hits[0] if hits else None


def _name_case(text: str) -> str:
    """SABEval's ``<class>`` substitution: 'heat capacity' -> 'HeatCapacity'."""
    return "".join(w[0].upper() + w[1:] if w and w[0].isalpha() else w
                   for w in text.split(" "))


def _instantiate(template: dict, derived_class: str,
                 add_params: dict) -> dict:
    """A tool for one derived class, with its additional parameters."""
    function = copy.deepcopy(template["tool_template"])
    function["parameters"]["properties"].update(add_params)
    function["parameters"]["required"] = (
        list(template["tool_template"]["parameters"]["required"])
        + list(add_params))
    function["name"] = function["name"].replace(
        "<class>", _name_case(derived_class))
    return json.loads(json.dumps(function).replace("<class>", derived_class))


def _additional_params(template: dict, count: int) -> dict:
    merged = {**template.get("raw_additional_params", {}),
              **template.get("additional_params", {})}
    return dict(itertools.islice(merged.items(), count))


def _process_key(template: dict, derived_class: str, count: int) -> str:
    params = (list(template["tool_template"]["parameters"]["required"])
              + list(_additional_params(template, count)))
    return (f"{template['tool_template_name']}|:|{derived_class}|:|"
            + ",".join(params))


def _sabeval_rows(root: Path, adds: tuple[int, ...],
                  render) -> list[tuple[str, str, int, str, str]]:
    """(request, tool, label, source, split) rows from SABEval."""
    templates = {}
    for path in sorted((root / "tool_templates").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        templates[rec["tool_template_name"]] = rec
    siblings = {}
    for path in sorted((root / "sibling_pairs").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        siblings[rec["tool_template_name"]] = rec["sibling_pairs"]
    out = []
    for add in adds:
        queries_dir = root / "queries" / f"add{add}"
        if not queries_dir.is_dir():
            continue
        queries = {}
        for path in sorted(queries_dir.glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            queries[list(rec.values())[0]["tool_template_name"]] = rec
        for name, template in templates.items():
            if name not in queries or name not in siblings:
                continue
            split = "dev" if int(_digest("sabeval:" + name), 16) % 5 == 0 \
                else "train"
            add_params = _additional_params(template, add)
            for query_class, tool_class in siblings[name]:
                key = _process_key(template, query_class, add)
                entry = queries[name].get(key)
                if not entry:
                    continue
                negative = _instantiate(template, tool_class, add_params)
                positive = _instantiate(template, query_class, add_params)
                for item in entry["queries"]:
                    out.append((item["query"], render(negative), 0,
                                "sabeval_sibling", split))
                    out.append((item["query"], render(positive), 1,
                                "sabeval_ground_truth", split))
    return out


# ---------------------------------------------------------------------------
# External synthetic near-misses (Hammer, arXiv 2410.04587)
# ---------------------------------------------------------------------------
def _hf_file(repo: str, filename: str, dest_dir: Path) -> Path | None:
    """Download one HF dataset file (cached) and keep a raw copy under
    ``dest_dir``. Offline after the first fetch. None when huggingface_hub is
    missing or the fetch fails, so the builder degrades to the BFCL corpus.
    """
    dest = dest_dir / Path(filename).name
    if dest.is_file():
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"skipping {repo}: huggingface_hub not installed", file=sys.stderr)
        return None
    try:
        cached = hf_hub_download(repo_id=repo, filename=filename,
                                 repo_type="dataset")
    except Exception as exc:  # network, auth, rate limit
        print(f"skipping {repo}/{filename}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(cached).read_bytes())
    return dest


def _load_tool_list(raw) -> list[dict]:
    """A tool list as dicts, from either upstream shape.

    xlam encodes the whole list as one JSON string; When2Call keeps a list
    whose elements are each a JSON string. Both are parsed here.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except ValueError:
                continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _xlam_pairs(render) -> list[tuple[str, str, int, str]]:
    """Should-abstain pairs from xlam-irrelevance-7.5k (Hammer, 2410.04587).

    The set is built by removing the ground-truth function from an
    xlam-function-calling catalogue and relabelling the row irrelevant, so
    every remaining listed tool is a near-miss negative for the query.
    """
    path = _hf_file(XLAM_REPO, XLAM_FILE, XLAM_DIR)
    if path is None:
        return []
    out = []
    for rec in json.loads(path.read_text(encoding="utf-8")):
        request = _clean(rec.get("query") or "")
        if not request:
            continue
        for tool in _load_tool_list(rec.get("tools")):
            span = _clean(render(tool))
            if span:
                out.append((request, span, 0, "synthetic_nearmiss_xlam"))
    return out


def _when2call_pairs(render) -> list[tuple[str, str, int, str]]:
    """Should-abstain pairs from When2Call MCQ, both near-miss constructions.

    ``cannot_answer`` rows list sibling tools that cannot serve the question.
    ``tool_call`` rows list the gold tool plus distractor siblings; every
    non-gold tool is an aligned-but-irrelevant negative. Both are the
    swap-in-a-sibling quadrant the abstention eval measures, which the
    xlam set (gold tool removed, so the remaining tools are foreign) is not.
    The gold tool and the ``request_for_info`` rows are skipped: the base
    corpus already supplies positives, and these would duplicate BFCL
    training requests.
    """
    path = _hf_file(WHEN2CALL_REPO, WHEN2CALL_MCQ, WHEN2CALL_DIR)
    if path is None:
        return []
    out = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        answer = rec.get("correct_answer")
        if answer not in ("cannot_answer", "tool_call"):
            continue
        request = _clean(rec.get("question") or "")
        if not request:
            continue
        target = rec.get("target_tool")
        gold = target.get("name") if isinstance(target, dict) else target
        for tool in _load_tool_list(rec.get("tools")):
            if answer == "tool_call" and gold and tool.get("name") == gold:
                continue
            span = _clean(render(tool))
            if span:
                out.append((request, span, 0,
                            "synthetic_nearmiss_when2call"))
    return out


# ---------------------------------------------------------------------------
# BFCL multi-turn: the session-conditioned evidence source (Apache-2.0)
# ---------------------------------------------------------------------------
def session_span(base: str, session: str) -> str:
    """Append the session channel to a rendered tool span.

    ``name: description | session: <prior tool calls / prior turn summary>``.
    A single-turn input renders ``| session: (none)`` so the span shape is
    identical on the two end-to-end eval sets, which carry no session.
    """
    return f"{_clean(base)} | session: {_clean(session) or SESSION_NONE}"


def _mt_catalogue() -> dict[str, list[dict]]:
    """Per-class tool lists from ``multi_turn_func_doc``, class-qualified.

    The doc files name tools bare (``mv``); the multi-turn ground truth and
    ``path`` name them class-qualified (``GorillaFileSystem.mv``), so every
    name is qualified here to match.
    """
    cats: dict[str, list[dict]] = {}
    for cls, fname in MT_CLASS_FILES.items():
        recs = _fetch(f"{BASE}/multi_turn_func_doc/{fname}.json",
                      MT_DOC_DIR / f"{fname}.json")
        cats[cls] = [dict(r, name=f"{cls}.{r['name']}") for r in recs]
    return cats


def _mt_answers(split: str) -> dict[str, list]:
    """Per-turn gold call lists for one multi-turn split."""
    recs = _fetch(f"{BASE}/possible_answer/BFCL_v4_{split}.json",
                  Path(f"/tmp/bfcl_v4_{split}_answer.json"))
    return {r["id"]: r["ground_truth"] for r in recs}


def _mt_session(prior_calls: list[str], prior_reqs: list[str]) -> str:
    """The session text for a turn: the calls and turns that came before it."""
    if not prior_calls and not prior_reqs:
        return SESSION_NONE
    turns = " ".join(prior_reqs)[:MT_PRIOR_TURNS_CHARS]
    return (f"prior calls: {', '.join(prior_calls) or 'none'}"
            f" | prior turns: {turns or 'none'}")


def multiturn_rows(split: str = MT_DEFAULT, render=render_tool) -> list[dict]:
    """(request, tool, label, session) rows from one BFCL multi-turn split.

    Each non-empty user turn becomes a labelled row per tool in the row's
    catalogue. The gold tools are the turn's own ``ground_truth`` calls; a
    turn whose gold function is excluded (``miss_func``) has no gold, so its
    whole catalogue is negative -- the abstention signal in session form.

    The session is the *prior* turns and prior gold calls only, so it never
    leaks the current turn's answer. Rows are split by conversation id, so a
    conversation lands wholly in train or dev.

    ``multi_turn_base``, ``multi_turn_miss_func`` and ``multi_turn_miss_param``
    re-ask the same conversations with different gold, so merging them gives
    the same (request, tool) pair both labels (1,345 pairs; see
    ``multiturn_rows_merged`` and ``reports/exp-pair-well-definedness.md``).
    This is the raw single-split builder; it enforces no constant label. One
    split is usually conflict-free, but two conversations can share a request
    string and disagree (``miss_param`` has one such pair). Callers that need
    the guarantee must go through ``multiturn_rows_merged``, which keeps only
    the constant-label pairs.
    """
    questions = _fetch(f"{BASE}/BFCL_v4_{split}.json",
                       Path(f"/tmp/bfcl_v4_{split}.json"))
    answers = _mt_answers(split)
    cats = _mt_catalogue()
    out: list[dict] = []
    for rec in questions:
        classes = [c for c in (rec.get("involved_classes") or []) if c in cats]
        tools = [t for c in classes for t in cats[c]]
        if not tools:
            continue
        names = {t["name"] for t in tools}
        gt = answers.get(rec["id"], [])
        split_tag = ("dev" if int(_digest(rec["id"]), 16) % 5 == 0
                     else "train")
        prior_calls: list[str] = []
        prior_reqs: list[str] = []
        gi = 0
        for turn in rec.get("question") or []:
            if not turn:
                continue
            request = _clean(turn[0].get("content", ""))
            golds: set[str] = set()
            if gi < len(gt):
                for call in gt[gi]:
                    match = MT_CALL.search(call)
                    if not match:
                        continue
                    for cls in classes:
                        qualified = f"{cls}.{match.group(1)}"
                        if qualified in names:
                            golds.add(qualified)
            gi += 1
            if not request:
                prior_calls += sorted(golds)
                continue
            session = _mt_session(prior_calls, prior_reqs)
            for tool in tools:
                label = 1 if tool["name"] in golds else 0
                source = ("mt_positive" if label
                          else "mt_distractor" if golds else "mt_miss_func")
                out.append({"request": request,
                            "tool": _clean(render(tool)), "label": label,
                            "source": source, "split": split_tag,
                            "session": session})
            prior_calls += sorted(golds)
            prior_reqs.append(request)
    return out


def multiturn_rows_merged(splits: Sequence[str] = MT_SPLITS,
                          render=render_tool) -> tuple[list[dict], dict[str, int]]:
    """(request, tool, label, session) rows merged across multi-turn splits.

    Merging splits breaks the (request, tool) unit: the three multi-turn
    splits re-ask the same conversations with different gold, so a pair can be
    gold in one split and a distractor or a missing-function negative in
    another. Across all three, **1,345 pairs** carry both labels and 3,604 rows
    sit on them. A training row whose label is a function of which split the
    row came from is a label the guard cannot recover from its input.

    The rule: the labelled unit is the (request, tool) pair, and its label must
    be **constant across every merged split**. A pair that disagrees is dropped
    whole -- every row on it, in every split -- rather than resolved by a
    tie-break, because the split is not part of the input the guard sees.

    Returns the kept rows plus counters: rows in, rows kept, rows dropped on a
    conflicting pair, distinct pairs, and conflicting pairs. Deterministic:
    ``splits`` is walked in order and every row of a constant pair is kept in
    place. ``multiturn_rows_merged((MT_DEFAULT,))`` reproduces
    ``multiturn_rows(MT_DEFAULT)`` exactly (that split has no intra-split
    conflict), which is what keeps the single-split corpus byte-identical.
    """
    all_rows: list[dict] = []
    for split in splits:
        all_rows += multiturn_rows(split, render)
    labels: dict[tuple[str, str], set[int]] = {}
    for row in all_rows:
        labels.setdefault((row["request"], row["tool"]), set()).add(
            row["label"])
    conflicting = {key for key, seen in labels.items() if len(seen) > 1}
    kept = [row for row in all_rows
            if (row["request"], row["tool"]) not in conflicting]
    stats = {
        "mt_merged_rows_in": len(all_rows),
        "mt_merged_rows_kept": len(kept),
        "mt_merged_rows_dropped_conflict": len(all_rows) - len(kept),
        "mt_merged_pairs": len(labels),
        "mt_merged_pairs_conflicting": len(conflicting),
    }
    return kept, stats


def _subsample(rows: list[tuple], cap: int) -> list[tuple]:
    """Deterministic content-hash ordering, truncated to ``cap`` (0 = all).

    Ordering by digest rather than file order makes the cap a stable,
    content-based subsample. Applied *after* the eval filter and dedup so the
    cap counts rows that actually enter the corpus.
    """
    ordered = sorted(rows, key=lambda r: _digest(r[0] + "\x00" + r[1]))
    return ordered if cap <= 0 else ordered[:cap]


# ---------------------------------------------------------------------------
# Fillability presence contrasts: minimal same-request twins.
#
# The gate span visibly renders `missing: <args>` yet the gate score does not
# move between the fillable and stripped sides of a minimal pair (median
# |delta| ~1e-5; reports/exp-fillability-diagnosis.md), because the training
# corpus carries no same-request minimal contrasts and pointwise
# cross-entropy never pressures the `missing:` clause. These twins supply
# that supervision: from REAL BFCL catalogues, the ONLY difference between
# the twins is required-argument presence (present vs stripped, same request
# topic, same tool). The strip is the guard-eval-v2 `no_tool_needed`
# construction, so the twins target exactly the no_tool_needed shape and the
# ACEBench-Special-incomplete (missing-arg) slice of infeasible.
# ---------------------------------------------------------------------------
CONTRAST_STRIPPED = "contrast_fillability_stripped"
_CONTRAST_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_CONTRAST_WS = re.compile(r"\s+")


def strip_required_values(request: str) -> str | None:
    """Remove the argument values a tool would fill from the request text.

    The same construction as the guard-eval-v2 `no_tool_needed` rows
    (`scripts/build_guard_eval.py::_strip_arguments`): numeric tokens go,
    whitespace collapses. None when the request carries no argument to strip.
    """
    stripped = _CONTRAST_WS.sub(
        " ", _CONTRAST_NUM.sub(" ", request)).strip()
    return stripped if stripped and stripped != request else None


def eligible_twin(request: str, tool: dict) -> str | None:
    """The stripped twin request, or None when this is not a presence contrast.

    Pure: the full request must fill every required arg (label 1 is honest)
    and the stripped twin must miss at least one (label 0 is honest, via
    `validate_fillability`). Otherwise the twin would teach a lie.
    """
    if not validate_fillability(tool, request)[0]:
        return None
    stripped = strip_required_values(request)
    if stripped is None:
        return None
    if validate_fillability(tool, stripped)[0]:
        return None
    return stripped


def _canon_tool(tool: dict) -> str:
    """Content key for a raw tool dict (eval-overlap checks)."""
    bare = tool.get("function") if isinstance(tool.get("function"), dict) \
        else tool
    return json.dumps(bare, sort_keys=True, ensure_ascii=False)


def _load_eval_index() -> dict[str, set]:
    """Every (request, tool) the eval sets score, plus v2 requests.

    Catalogue files are indexed by (request, canonical-tool); the committed
    pair corpus by (request, stored span). guard-eval-v2 is indexed both ways
    so a zero (request, tool) overlap can be verified exactly.
    """
    cat_pairs: set[tuple[str, str]] = set()
    for name in ("calls.jsonl", "irrelevance_eval.jsonl", "irrelevance.jsonl"):
        path = BFCL / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(
                json.loads(line).get("text") or "")
            request = _clean(request)
            for tool in tools or []:
                cat_pairs.add((request, _canon_tool(tool)))
    v2_path = BFCL / "guard-eval-v2.jsonl"
    v2_pairs: set[tuple[str, str]] = set()
    v2_requests: set[str] = set()
    if v2_path.is_file():
        for line in v2_path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            request = _clean(rec.get("request", ""))
            v2_requests.add(request)
            v2_pairs.add((request, _canon_tool(rec.get("tool") or {})))
    corpus_pairs: set[tuple[str, str]] = set()
    if OUT.is_file():
        for line in OUT.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            corpus_pairs.add((_clean(rec.get("request", "")),
                              _clean(str(rec.get("tool") or ""))))
    for name in ("pairs-ace.jsonl", "pairs-nearmiss.jsonl"):
        path = BFCL / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            tool = rec.get("tool")
            span = _clean(render_tool(tool)) if isinstance(tool, dict) \
                else _clean(str(tool or ""))
            corpus_pairs.add((_clean(rec.get("request", "")), span))
    return {"catalogue": cat_pairs, "v2_pairs": v2_pairs,
            "v2_requests": v2_requests, "corpus": corpus_pairs}


def overlap_decision(request: str, canon: str, v1span: str,
                     index: dict[str, set]) -> str | None:
    """The drop reason for a stripped twin, or None when it is clean.

    Pure: every eval-overlap rule in one place so the corpus builder and the
    tests read the same contract. `request` is the STRIPPED twin request.
    """
    if request in index["v2_requests"]:
        return "eval_request_v2"
    if (request, canon) in index["v2_pairs"]:
        return "eval_pair_v2"
    if (request, canon) in index["catalogue"]:
        return "eval_pair_catalogue"
    if (request, v1span) in index["corpus"]:
        return "training_pair_corpus"
    return None


def contrast_candidates() -> list[tuple[str, str, dict]]:
    """Eligible (full request, stripped request, gold tool) triples.

    Same gold walk as `build()`: every called tool of every non-eval request.
    Content-hash ordered (no RNG). Promo to rows happens in
    `build_gate_corpus`, which additionally requires the twin's base positive
    to exist and the stripped side to clear every eval-overlap rule.
    """
    questions, answers = _sources()
    blocked = _eval_requests()
    triples = []
    for rec in questions:
        request = rec["question"][0][0]["content"]
        if _clean(request) in blocked:
            continue
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        for tool in rec["function"]:
            if tool["name"] not in called:
                continue
            stripped = eligible_twin(request, tool)
            if stripped is None:
                continue
            if _clean(stripped) in blocked:
                continue
            triples.append((_clean(request), _clean(stripped), tool))
    triples.sort(key=lambda t: _digest(t[0] + "\x00" + _canon_tool(t[2])))
    return triples


def twin_split(full_request: str, v1span: str) -> str:
    """The twin's split: its full side's split, so twins never straddle."""
    return _split_of(full_request, v1span)


def build_gate_corpus(contrast: bool = False,
                      cap: int = 0) -> tuple[list[dict], dict]:
    """Gate-span training corpus: base rows plus presence-contrast twins.

    The base walk reproduces `scripts/exp_span_redesign.py::build_rows("v3")`
    verbatim (v1-keyed dedup and split; only the stored tool string changes),
    rendering the fillability gate span (`render_tool_fillability`, verified
    string-identical to that harness's `span_v3`). With `contrast`, each
    eligible gold additionally contributes its stripped twin
    (`contrast_fillability_stripped`, label 0) carrying `twin_key` — the
    digest of its full side's (request, v1 span) — so the trainer can press
    the twins apart. Every row carries `key` (its own pair digest); twin rows
    carry `twin_key` too. Twins whose stripped side hits any eval-overlap
    rule, or whose full side has no base positive, are dropped and counted.
    `cap` bounds the twin count (0 = all), post-filter, in hash order.
    """
    stats: dict[str, int] = {
        "bfcl_positives": 0, "bfcl_distractor_negatives": 0,
        "bfcl_distractor_aligned": 0, "bfcl_foreign_negatives": 0,
        "bfcl_irrelevance_negatives": 0,
        "contrast_twins": 0, "contrast_capped": 0,
        "dropped_eval_request": 0, "dropped_duplicate": 0,
        "dropped_twin_eval_overlap": 0, "dropped_twin_no_base": 0}
    blocked = _eval_requests()
    seen: set[tuple[str, str]] = set()
    base_keys: dict[tuple[str, str], str] = {}
    rows: list[dict] = []

    def add(request: str, tool: dict, label: int, counter: str,
            span_fn, split: str | None = None) -> bool:
        request = _clean(request)
        if not request:
            return False
        if request in blocked:
            stats["dropped_eval_request"] += 1
            return False
        v1_span = _clean(render_tool(tool))
        key = (request, v1_span)
        if key in seen:
            stats["dropped_duplicate"] += 1
            return False
        seen.add(key)
        tool_span = _clean(span_fn(tool, request))
        if not tool_span:
            return False
        rows.append({"request": request, "tool": tool_span, "label": label,
                     "source": counter,
                     "split": split or _split_of(request, v1_span),
                     "key": _digest(request + "\x00" + v1_span)})
        stats[counter] += 1
        return True

    gate_span = render_tool_fillability
    questions, answers = _sources()
    matched: list[tuple[str, dict]] = []
    for rec in questions:
        request = rec["question"][0][0]["content"]
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        gold_signatures = {_signature(t) for t in rec["function"]
                           if t["name"] in called}
        gold_signatures.discard(())
        for tool in rec["function"]:
            if tool["name"] in called:
                matched.append((request, tool))
            elif _signature(tool) and _signature(tool) in gold_signatures:
                add(request, tool, 0, "bfcl_distractor_aligned", gate_span)
            else:
                add(request, tool, 0, "bfcl_distractor_negatives", gate_span)

    span = len(matched) // 2
    for i, (request, tool) in enumerate(matched):
        kept = add(request, tool, 1, "bfcl_positives", gate_span)
        v1 = _clean(render_tool(tool))
        if kept:
            base_keys[(_clean(request), v1)] = twin_split(
                _clean(request), v1)
        for step in range(1, 5):
            foreign = matched[(i + span * step) % len(matched)][1]
            if foreign["name"].split(":")[0] != tool["name"].split(":")[0]:
                add(request, foreign, 0, "bfcl_foreign_negatives", gate_span)
                break

    irr = BFCL / "irrelevance.jsonl"
    if irr.is_file():
        for line in irr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(json.loads(line)["text"])
            for tool in tools or []:
                add(request, tool, 0, "bfcl_irrelevance_negatives", gate_span)

    if contrast:
        index = _load_eval_index()
        kept_twins = 0
        for full, stripped, tool in contrast_candidates():
            v1 = _clean(render_tool(tool))
            if (full, v1) not in base_keys:
                stats["dropped_twin_no_base"] += 1
                continue
            reason = overlap_decision(stripped, _canon_tool(tool), v1, index)
            if reason is not None:
                stats["dropped_twin_eval_overlap"] += 1
                continue
            if cap > 0 and kept_twins >= cap:
                stats["contrast_capped"] += 1
                continue
            twin_key = _digest(full + "\x00" + v1)
            span_text = _clean(gate_span(tool, stripped))
            if not span_text or stripped in blocked:
                stats["dropped_twin_eval_overlap"] += 1
                continue
            if (stripped, v1) in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add((stripped, v1))
            rows.append({"request": stripped, "tool": span_text, "label": 0,
                         "source": CONTRAST_STRIPPED,
                         "split": base_keys[(full, v1)],
                         "key": _digest(stripped + "\x00" + v1),
                         "twin_key": twin_key, "twin_request": full})
            stats["contrast_twins"] += 1
            kept_twins += 1
    return rows, stats


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build(rendering: str = "v1",
          sabeval: str = "none",
          mask_name: bool = False,
          nearmiss: str = "none") -> tuple[list[dict], dict[str, int]]:
    base_render = (partial(render_tool, parameters=True) if rendering == "v2"
                   else render_tool)

    def render(tool: dict) -> str:
        """The tool span. ``mask_name`` blanks the name (Hammer, ICLR 2025).

        Function masking removes the name-matching shortcut, so the model has
        to judge irrelevance from the description and parameters instead of
        from a surface match on the tool's identifier.
        """
        text = base_render(tool)
        if not mask_name:
            return text
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = (fn or {}).get("name", "")
        return text.replace(f"{name}:", "<tool>:", 1) if name else text
    questions, answers = _sources()
    blocked = _eval_requests()
    stats: dict[str, int] = {
        "bfcl_positives": 0, "bfcl_distractor_negatives": 0,
        "bfcl_distractor_aligned": 0, "bfcl_foreign_negatives": 0,
        "bfcl_irrelevance_negatives": 0, "sabeval_sibling": 0,
        "sabeval_ground_truth": 0, "synthetic_nearmiss_xlam": 0,
        "synthetic_nearmiss_when2call": 0,
        "dropped_eval_request": 0, "dropped_duplicate": 0}
    seen: dict[tuple[str, str], int] = {}
    rows: list[dict] = []

    def add(request: str, tool: str, label: int, counter: str,
            split: str | None = None) -> None:
        request, tool = _clean(request), _clean(tool)
        if not request or not tool:
            return
        if request in blocked:
            stats["dropped_eval_request"] += 1
            return
        if (request, tool) in seen:
            stats["dropped_duplicate"] += 1
            return
        seen[(request, tool)] = label
        rows.append({"request": request, "tool": tool, "label": label,
                     "source": counter,
                     "split": split or _split_of(request, tool)})
        stats[counter] += 1

    matched: list[tuple[str, dict]] = []
    for rec in questions:
        request = rec["question"][0][0]["content"]
        called = {next(iter(g)) for g in answers.get(rec["id"], []) if g}
        gold_signatures = {_signature(t) for t in rec["function"]
                           if t["name"] in called}
        gold_signatures.discard(())
        for tool in rec["function"]:
            if tool["name"] in called:
                matched.append((request, tool))
            elif _signature(tool) and _signature(tool) in gold_signatures:
                add(request, render(tool), 0, "bfcl_distractor_aligned")
            else:
                add(request, render(tool), 0, "bfcl_distractor_negatives")

    span = len(matched) // 2
    for i, (request, tool) in enumerate(matched):
        add(request, render(tool), 1, "bfcl_positives")
        for step in range(1, 5):
            foreign = matched[(i + span * step) % len(matched)][1]
            if foreign["name"].split(":")[0] != tool["name"].split(":")[0]:
                add(request, render(foreign), 0, "bfcl_foreign_negatives")
                break

    # The abstention signal: a catalogue where nothing matches. Safe to pool
    # now that the foreign negatives dominate and carry the positives' length
    # distribution; on their own they were what taught the length shortcut.
    # The dev rows are dropped by `blocked` above, so they stay unseen.
    irr = BFCL / "irrelevance.jsonl"
    if irr.is_file():
        for line in irr.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            tools, request = parse_catalogue_raw(json.loads(line)["text"])
            for tool in tools or []:
                add(request, render(tool), 0, "bfcl_irrelevance_negatives")

    adds = {"none": (), "add0": (0,), "all": (0, 1, 2, 3, 4)}[sabeval]
    root = _sabeval_root() if adds else None
    if root is not None:
        for request, tool, label, source, split in _sabeval_rows(root, adds,
                                                                 render):
            add(request, tool, label, source, split)

    # External synthetic near-miss augmentation (Hammer, arXiv 2410.04587).
    # Routed through `add`, so the request-level eval filter and the global
    # (request, tool) dedup apply exactly as they do to the BFCL rows; the cap
    # counts rows that actually enter the corpus (post-dedup), in a
    # deterministic content-hash order.
    def add_capped(pairs: list[tuple], cap: int) -> None:
        added = 0
        for request, tool, label, source in _subsample(pairs, 0):
            before = stats[source]
            add(request, tool, label, source)
            if stats[source] > before:
                added += 1
                if cap > 0 and added >= cap:
                    break

    if nearmiss in ("xlam", "both"):
        add_capped(_xlam_pairs(render), XLAM_CAP)
    if nearmiss in ("when2call", "both"):
        add_capped(_when2call_pairs(render), WHEN2CALL_CAP)
    return rows, stats


def build_session(mt_split: str = MT_DEFAULT,
                  mt_splits: Sequence[str] | None = None
                  ) -> tuple[list[dict], dict]:
    """The shipped corpus plus the multi-turn session channel.

    Every row's ``tool`` becomes ``<v1 span> | session: <...>``. The base rows
    (and, at eval time, the two end-to-end sets) carry ``session: (none)``;
    the multi-turn rows carry the prior turns and prior gold calls. The
    (request, tool) label is unchanged from the shipped corpus for base rows
    and from the per-turn ground truth for multi-turn rows.

    The request-level eval filter is applied to the multi-turn rows too, so a
    session request that also appears in an eval set cannot train the guard.

    ``mt_split`` builds one multi-turn split (the default, byte-identical to
    the shipped session corpus). ``mt_splits`` merges several through
    ``multiturn_rows_merged``, which drops every (request, tool) pair whose
    label disagrees across the merged splits; its counters are folded into the
    returned stats.
    """
    base_rows, base_stats = build(rendering="v1")
    # Fresh counters: ``base_rows`` already carry their tags, so counting them
    # again here (rather than reusing build()'s) keeps every source counted
    # exactly once.
    stats: dict[str, int] = {name: 0 for name in base_stats}
    blocked = _eval_requests()
    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []

    def add(request: str, tool: str, label: int, source: str,
            session: str, split: str) -> None:
        request, tool = _clean(request), _clean(tool)
        if not request or not tool:
            return
        if request in blocked:
            stats["dropped_eval_request"] = stats.get(
                "dropped_eval_request", 0) + 1
            return
        if (request, tool) in seen:
            stats["dropped_duplicate"] = stats.get("dropped_duplicate", 0) + 1
            return
        seen.add((request, tool))
        rows.append({"request": request, "tool": tool, "label": label,
                     "source": source, "split": split,
                     "session": _clean(session) or SESSION_NONE})
        stats[source] = stats.get(source, 0) + 1

    if mt_splits is None:
        mt_rows, mt_stats = multiturn_rows(mt_split), {}
    else:
        mt_rows, mt_stats = multiturn_rows_merged(mt_splits)
    for row in base_rows:
        add(row["request"], session_span(row["tool"], ""), row["label"],
            row["source"], "", row["split"])
    for row in mt_rows:
        add(row["request"], session_span(row["tool"], row["session"]),
            row["label"], row["source"], row["session"], row["split"])
    stats.update(mt_stats)
    return rows, stats


def _parse_mt_splits(value: str) -> list[str]:
    """A comma-separated split list, or ``all`` for every multi-turn split."""
    if value.strip() == "all":
        return list(MT_SPLITS)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise SystemExit("--mt-splits is empty; give split names or 'all'")
    unknown = [p for p in parts if p not in MT_SPLITS]
    if unknown:
        raise SystemExit(f"unknown multi-turn split(s) {unknown}; "
                         f"choose from {MT_SPLITS}")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendering", choices=("v1", "v2"), default="v1",
                        help="v1 is the shipped name+description span; v2 "
                             "adds the parameter schema")
    parser.add_argument("--sabeval", choices=("none", "add0", "all"),
                        default="none")
    parser.add_argument("--mask-name", action="store_true",
                        help="blank the tool name in the span (function "
                             "masking; Hammer, ICLR 2025)")
    parser.add_argument("--nearmiss", choices=("none", "xlam", "when2call",
                                               "both"), default="none",
                        help="augment with external synthetic near-miss "
                             "negatives (Hammer, arXiv 2410.04587)")
    parser.add_argument("--session", action="store_true",
                        help="append the BFCL multi-turn session channel to "
                             "every span and mix in the multi-turn pairs "
                             "(single-turn rows render '| session: (none)')")
    parser.add_argument("--mt-split", choices=MT_SPLITS, default=MT_DEFAULT,
                        help="the multi-turn split used by --session")
    parser.add_argument("--mt-splits", default=None,
                        help="comma-separated multi-turn splits to merge, or "
                             "'all'; keeps only (request, tool) pairs whose "
                             "label is constant across the merged splits and "
                             "reports the rows dropped on conflicting pairs")
    parser.add_argument("--gate-span", action="store_true",
                        help="render the fillability gate span "
                             "(render_tool_fillability, string-identical to "
                             "exp_span_redesign span_v3) instead of v1; the "
                             "corpus a gate candidate trains on")
    parser.add_argument("--contrast", type=int, default=0,
                        help="append up to N presence-contrast stripped twins "
                             "(requires --gate-span; 0 = off, the default "
                             "keeps the shipped corpus byte-identical)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.contrast and not args.gate_span:
        raise SystemExit("--contrast requires --gate-span")
    if args.gate_span and args.session:
        raise SystemExit("--gate-span cannot combine with --session")
    if args.gate_span:
        rows, stats = build_gate_corpus(contrast=args.contrast > 0,
                                        cap=args.contrast)
        payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        for name, value in stats.items():
            print(f"{name}: {value}")
        print(f"total rows: {len(rows)}")
        print(f"sha256: {digest}")
        print(f"written to {args.out}")
        return 0

    if args.session:
        if args.mt_splits:
            rows, stats = build_session(
                mt_splits=_parse_mt_splits(args.mt_splits))
        else:
            rows, stats = build_session(mt_split=args.mt_split)
    else:
        rows, stats = build(rendering=args.rendering, sabeval=args.sabeval,
                            mask_name=args.mask_name, nearmiss=args.nearmiss)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    fp_rows = _write_fp_eval(_fp_eval_rows())
    labels = {r["label"] for r in rows}
    splits = {r["split"] for r in rows}
    for name, value in stats.items():
        print(f"{name}: {value}")
    print(f"total rows: {len(rows)}  labels: {sorted(labels)}  "
          f"splits: {sorted(splits)}")
    print(f"sha256: {digest}")
    print(f"written to {args.out}")
    print(f"abstention eval: {fp_rows} rows -> {FP_EVAL.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
